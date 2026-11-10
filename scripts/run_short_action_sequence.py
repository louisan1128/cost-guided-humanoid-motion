#!/usr/bin/env python3
"""V3 horizon ablation; reuse the certified sequence physics/scoring runner."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_state_action_sequence as physics
from pilot_direct_action_bc import sha, write
from humanoid_imitation.planning.short_action_sequence import (
    ShortActionSequenceNet, reconstruction_losses,
)
from humanoid_imitation.planning.temporal_action_sequence import TemporalActionSequenceNet, independent_noise
from humanoid_imitation.planning.state_action_sequence import (
    alpha_schedule, noisy_action, sample_sequence, PlanExecutionBuffer,
)


def make_model(checkpoint):
    cls = TemporalActionSequenceNet if checkpoint.get('architecture') == 'temporal_tokens' else ShortActionSequenceNet
    return cls(
        checkpoint["context_dim"], checkpoint["horizon"],
        checkpoint["action_horizon"], checkpoint["method"],
    )


class ShortActionPolicy(physics.SequencePolicy):
    def __init__(self, path, seed=0, guide=0., guide_target=0., unit_gradient=False,
                 execute_steps=1, guide_mode="lateral"):
        c = torch.load(path, map_location="cuda", weights_only=False)
        if execute_steps > c["action_horizon"]:
            raise ValueError("Cannot execute masked action tokens")
        self.c = c
        self.stats = {k: v.cuda() for k, v in c["stats"].items()}
        self.h = c["horizon"]
        self.joint = True
        self.token = 99
        self.hist, self.actions = [], []
        self.model = make_model(c).cuda().eval()
        self.model.load_state_dict(c["model"])
        self.alpha = alpha_schedule().cuda()
        self.rng = torch.Generator(device="cuda").manual_seed(seed)
        self.guide, self.guide_target = guide, guide_target
        self.last, self.unit_gradient = None, unit_gradient
        self.execution = PlanExecutionBuffer(execute_steps)
        self.guide_mode = guide_mode
        self.plan_age = self.queries = 0


def train(a):
    physics.check(a)
    a.out_dir.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    torch.set_num_threads(4)
    torch.manual_seed(a.seed)
    d = torch.load(a.data, weights_only=False)
    if not d["joint"] or d["horizon"] != 30:
        raise ValueError("This preregistered ablation requires joint H30 data")
    stats = {k: v.cuda() for k, v in d["stats"].items()}
    h = d["horizon"]

    def normalized(split):
        x = d[split]["context"].cuda()
        y = d[split]["target"].flatten(1).cuda()
        return ((x - stats["context_mean"]) / stats["context_std"]).clamp(-10, 10), (y - stats["target_mean"]) / stats["target_std"]

    x, y = normalized("train")
    vx, vy = normalized("validation")
    g = torch.Generator(device="cuda").manual_seed(4405)
    ids = torch.randperm(len(vx), generator=g, device="cuda")[:2048]
    vx, vy = vx[ids], vy[ids]
    c = dict(stats=d["stats"], method=a.method, horizon=h,
             action_horizon=a.action_horizon, joint=True, token_dim=99,
             context_dim=x.shape[-1], seed=a.seed, history=4, execute=1,
             architecture=a.architecture, teacher_sha256=sha(physics.TEACHER))
    model = make_model(c).cuda()
    y = y * model.valid_output
    vy = vy * model.valid_output
    ema = copy.deepcopy(model).eval()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.updates, eta_min=3e-5)
    alpha = alpha_schedule().cuda()
    noise = torch.randn(vy.shape, generator=g, device="cuda")
    best, curve = float("inf"), []
    for step in range(1, a.updates + 1):
        ix = torch.randint(len(x), (512,), device="cuda")
        bx, by = x[ix], y[ix]
        if a.method == "mlp":
            pred = model(bx)
        else:
            shape = (len(bx), h+a.action_horizon) if a.architecture == 'temporal_tokens' else (len(bx),)
            t = torch.randint(100, shape, device="cuda")
            t[:len(t)//2] = torch.randint(75, 100, t[:len(t)//2].shape, device="cuda")
            noisy = independent_noise(by, torch.randn_like(by), t, alpha, a.action_horizon) if a.architecture == 'temporal_tokens' else noisy_action(by, torch.randn_like(by), t, alpha)
            pred = model(bx, noisy, t)
        al, sl = reconstruction_losses(
            pred.reshape(-1, h, 99), by.reshape(-1, h, 99),
            a.action_horizon, a.first_fraction,
        )
        loss = al + a.state_weight * sl
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training objective")
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step()
        schedule.step()
        with torch.no_grad():
            for ep, p in zip(ema.parameters(), model.parameters()):
                ep.lerp_(p, .005)
        if step % 500 == 0 or step == a.updates:
            with torch.no_grad():
                out = torch.cat([sample_sequence(ema, vx[k:k+256], alpha, noise[k:k+256]) for k in range(0, len(vx), 256)])
                error = (out - vy).reshape(-1, h, 99)
                score = float(error[:, 0, :28].square().mean())
                physical = error * stats["target_std"].reshape(h, 99)
                row = dict(update=step, loss=float(loss), action_loss=float(al), state_loss=float(sl),
                           validation_first_action_mse=score,
                           valid_action_rmse_rad=float(physical[:, :a.action_horizon, :28].square().mean().sqrt()),
                           validation_state_mse=float(error[:, :, 28:].square().mean()))
            curve.append(row)
            print(json.dumps(row), flush=True)
            if score < best:
                best = score
                torch.save(dict(c, model={k: v.cpu().clone() for k, v in ema.state_dict().items()}, update=step), a.out_dir / "model.pt")
    write(a.out_dir / "learning_curve.json", curve)
    write(a.out_dir / "result.json", dict(
        status="complete", method=a.method, architecture=a.architecture, action_horizon=a.action_horizon,
        state_horizon=h, updates=a.updates, seed=a.seed,
        train_windows=len(x), validation_windows=len(vx),
        first_fraction=a.first_fraction, state_weight=a.state_weight,
        parameters=sum(p.numel() for p in model.parameters()),
        wall_s=time.monotonic()-start, best_validation_first_action_mse=best,
        dataset_sha256=sha(a.data), model_sha256=sha(a.out_dir / "model.pt"),
        optimizer_resume=False, script_sha256=sha(__file__),
        model_source_sha256=sha(ROOT / "src/humanoid_imitation/planning/short_action_sequence.py"),
        temporal_source_sha256=sha(ROOT / "src/humanoid_imitation/planning/temporal_action_sequence.py"),
    ))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["train", "evaluate", "probe"])
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--policy", type=Path)
    p.add_argument("--method", choices=["mlp", "diffusion"], default="mlp")
    p.add_argument("--action-horizon", type=int, default=8)
    p.add_argument("--architecture", choices=['flat_masked_short_action', 'temporal_tokens'], default='flat_masked_short_action')
    p.add_argument("--updates", type=int, default=8000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--first-fraction", type=float, default=16/45)
    p.add_argument("--state-weight", type=float, default=.001)
    p.add_argument("--cohort", choices=["train", "validation"], default="validation")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--guide", type=float, default=0.)
    p.add_argument("--guide-target", type=float, default=0.)
    p.add_argument("--guide-start-seconds", type=float, default=2.)
    p.add_argument("--guide-mode", choices=["lateral", "heading"], default="heading")
    p.add_argument("--unit-gradient", action="store_true")
    p.add_argument("--probe-steps", type=int, default=120)
    p.add_argument("--execute-steps", type=int, default=1)
    a = p.parse_args()
    for k in ["root", "out_dir", "source", "data", "policy"]:
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    if a.mode == "train":
        train(a)
    else:
        physics.SequencePolicy = ShortActionPolicy
        physics.simulate(a)
        r = json.loads((a.out_dir / "result.json").read_text())
        r.update(runner=str(Path(__file__).resolve()), runner_sha256=sha(__file__),
                 model_source_sha256=sha(ROOT / "src/humanoid_imitation/planning/short_action_sequence.py"))
        write(a.out_dir / "result.json", r)


if __name__ == "__main__":
    main()
