#!/usr/bin/env python3
"""Fixed world-heading diagnostic, reusing the V3 physical execution contract."""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_state_action_sequence as physics
from run_short_action_sequence import ShortActionPolicy
from pilot_direct_action_bc import sha, write
from humanoid_imitation.planning.state_action_sequence import sample_sequence, yaw


def fixed_heading_cost(sequence, anchor_yaw, desired_world_yaw, nominal_endpoint):
    end = sequence[:, -1]
    local_yaw = torch.atan2(end[:, 32], end[:, 31])
    angle = local_yaw + anchor_yaw - desired_world_yaw
    return 1 - angle.cos() + .25 * (end[:, 28] - nominal_endpoint[:, 28]).square()


class FixedWorldPolicy(ShortActionPolicy):
    def __init__(self, *args, env, **kwargs):
        super().__init__(*args, **kwargs)
        if self.execution.execute_steps != 1:
            raise ValueError("This diagnostic requires E1")
        self.env = env
        self.desired_world_yaw = None
        self.guidance_onset = None

    def __call__(self, obs):
        self.hist.append(obs.detach().clone())
        self.hist = self.hist[-4:]
        history = [self.hist[0]] * (4-len(self.hist)) + self.hist
        past = [torch.zeros(len(obs), 28, device=obs.device)] * (3-len(self.actions)) + self.actions
        context = torch.cat([*history, *past], -1)
        context = ((context-self.stats['context_mean'])/self.stats['context_std']).clamp(-10, 10)
        noise = torch.randn(len(obs), self.h*99, device=obs.device, generator=self.rng)
        nominal_endpoint = None
        if self.guide:
            anchor = yaw(self.env._engine.get_root_rot(self.env._get_char_id())).detach()
            with torch.no_grad():
                z = sample_sequence(self.model, context, self.alpha, noise, 16)
                nominal_endpoint = (z*self.stats['target_std']+self.stats['target_mean']).reshape(len(obs), self.h, 99)[:, -1].detach()
            if self.desired_world_yaw is None:
                self.desired_world_yaw = (anchor + torch.atan2(nominal_endpoint[:, 32], nominal_endpoint[:, 31]) + math.radians(self.guide_target)).detach()
                self.guidance_onset = dict(anchor_world_yaw=anchor.cpu().tolist(), desired_world_yaw=self.desired_world_yaw.cpu().tolist())

        def cost(z):
            seq = (z*self.stats['target_std']+self.stats['target_mean']).reshape(len(obs), self.h, 99)
            return fixed_heading_cost(seq, anchor, self.desired_world_yaw, nominal_endpoint)

        with torch.no_grad():
            z = sample_sequence(self.model, context, self.alpha, noise, 16, cost if self.guide else None, self.guide, self.unit_gradient)
            self.last = (z*self.stats['target_std']+self.stats['target_mean']).reshape(len(obs), self.h, 99).detach()
        self.queries += 1
        self.plan_age = 0
        action = self.last[:, 0, :28]
        self.actions = (self.actions + [action.detach().clone()])[-3:]
        return action


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--policy', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--guide', type=float, default=0.)
    p.add_argument('--guide-target', type=float, default=0.)
    p.add_argument('--seed', type=int, default=4424)
    a = p.parse_args()
    for k in ['root', 'source', 'policy', 'out_dir']:
        setattr(a, k, getattr(a, k).resolve())
    a.mode, a.cohort, a.limit = 'probe', 'validation', 12
    a.probe_steps, a.execute_steps = 120, 1
    a.guide_mode, a.unit_gradient, a.guide_start_seconds = 'heading', False, 2.
    holder = {}
    original_init = physics.init_env

    def init_with_observed_env(*args, **kwargs):
        env, agent = original_init(*args, **kwargs)
        holder['env'] = env
        return env, agent

    def make_policy(*args, **kwargs):
        policy = FixedWorldPolicy(*args, env=holder['env'], **kwargs)
        holder['policy'] = policy
        return policy

    # This standalone diagnostic shares scoring/state/reset code with the prior study.
    physics.init_env = init_with_observed_env
    physics.SequencePolicy = make_policy
    physics.simulate(a)
    result = json.loads((a.out_dir/'result.json').read_text())
    result.update(guidance='fixed world heading from same-noise nominal prediction at t2; no per-query target reset',
                  guidance_onset=holder['policy'].guidance_onset, runner_sha256=sha(__file__))
    write(a.out_dir/'result.json', result)


if __name__ == '__main__':
    main()
