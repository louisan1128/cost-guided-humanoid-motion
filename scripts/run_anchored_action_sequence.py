#!/usr/bin/env python3
"""Bounded supervised sequence correction with frozen V12 action priors."""
import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import run_short_action_sequence as core
from pilot_direct_action_bc import sha, write
from humanoid_imitation.planning.direct_action import DirectMLP, ActionDenoiser
from humanoid_imitation.planning.anchored_action_sequence import AnchoredActionSequenceNet


def make_anchored(c):
    path = Path(c.get('prior_path', ROOT/f"outputs/direct_action_study_v12/final/{c['method']}/seed0/model.pt"))
    prior_c = torch.load(path, map_location='cpu', weights_only=False)
    if c.get('prior_sha256') is not None and sha(path) != c['prior_sha256']:
        raise ValueError('Frozen prior changed')
    c.update(prior_path=str(path), prior_sha256=sha(path), correction_limit_rad=.03)
    checks = [('obs_mean', c['stats']['context_mean'][432:576]),
              ('obs_std', c['stats']['context_std'][432:576]),
              ('action_mean', c['stats']['target_mean'][:28]),
              ('action_std', c['stats']['target_std'][:28])]
    if not all(torch.equal(prior_c['stats'][k].cpu(), v.cpu()) for k, v in checks):
        raise ValueError('Prior/sequence normalization mismatch')
    cls = DirectMLP if c['method'] == 'mlp' else ActionDenoiser
    prior = cls()
    prior.load_state_dict(prior_c['model'])
    return AnchoredActionSequenceNet(prior, c['method'], c['stats']['target_std'][:28],
                                     c['horizon'], c['action_horizon'], c['correction_limit_rad'])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('mode', choices=['train', 'evaluate', 'probe'])
    for key in ['root', 'source', 'out-dir', 'data', 'policy']:
        p.add_argument('--'+key, type=Path, required=key in ['root', 'out-dir'])
    p.add_argument('--method', choices=['mlp', 'diffusion'], default='mlp')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--guide', type=float, default=0.)
    p.add_argument('--guide-target', type=float, default=0.)
    p.add_argument('--unit-gradient', action='store_true')
    a = p.parse_args()
    for k in ['root', 'source', 'out_dir', 'data', 'policy']:
        if getattr(a, k) is not None:
            setattr(a, k, getattr(a, k).resolve())
    a.architecture, a.action_horizon, a.updates = 'prior_anchored', 8, 8000
    a.state_weight, a.first_fraction = .001, 16/45
    a.cohort, a.limit, a.execute_steps = 'validation', 0, 1
    a.guide_mode, a.guide_start_seconds = 'heading', 2.
    a.probe_steps = 120
    core.make_model = make_anchored
    if a.mode == 'train':
        core.train(a)
        c = torch.load(a.out_dir/'model.pt', map_location='cpu', weights_only=False)
        prior_c = torch.load(c['prior_path'], map_location='cpu', weights_only=False)
        equal = all(torch.equal(c['model']['prior.'+k], v) for k, v in prior_c['model'].items())
        if not equal:
            raise ValueError('Prior tensor changed during sequence training')
        result = json.loads((a.out_dir/'result.json').read_text())
        result.update(prior_tensor_exact=equal, prior_path=c['prior_path'], prior_sha256=c['prior_sha256'],
                      correction_limit_rad=.03, supervision='teacher joint sequences; frozen prior; no residual RL')
    else:
        if a.mode == 'probe':
            from probe_short_action_guidance import FixedWorldPolicy
            a.limit = 12
            holder, original = {}, core.physics.init_env
            def retain_env(*args, **kwargs):
                env, agent = original(*args, **kwargs)
                holder['env'] = env
                return env, agent
            def policy(*args, **kwargs):
                holder['policy'] = FixedWorldPolicy(*args, env=holder['env'], **kwargs)
                return holder['policy']
            core.physics.init_env, core.physics.SequencePolicy = retain_env, policy
        else:
            core.physics.SequencePolicy = core.ShortActionPolicy
        core.physics.simulate(a)
        result = json.loads((a.out_dir/'result.json').read_text())
        if a.mode == 'probe':
            result.update(guidance_onset=holder['policy'].guidance_onset, guidance='fixed world heading')
    result.update(runner_sha256=sha(__file__), shared_training_source_sha256=sha(ROOT/'scripts/run_short_action_sequence.py'),
                  anchored_source_sha256=sha(ROOT/'src/humanoid_imitation/planning/anchored_action_sequence.py'))
    write(a.out_dir/'result.json', result)


if __name__ == '__main__':
    main()
