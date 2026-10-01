#!/usr/bin/env python3
"""Measure physical and reference root XY speed during one PPO tracking episode."""
import argparse
import json
from pathlib import Path

from evaluate_mixamo_tracker import MIMICKIT
import envs.base_env as base_env
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.util as mimic_util
import torch
import os


def measure(env_config: Path, model_file: Path, seed: int, bin_seconds: float) -> dict:
    env_config, model_file = env_config.resolve(), model_file.resolve()
    os.chdir(MIMICKIT)
    device = 'cuda:0'
    mp_util.init(0, 1, device, 6341)
    mimic_util.set_rand_seed(seed)
    env = env_builder.build_env(str(env_config), 'data/engines/isaac_lab_engine.yaml',
                                1, device, visualize=False)
    agent = agent_builder.build_agent('data/agents/deepmimic_humanoid_ppo_agent.yaml', env, device)
    agent.load(str(model_file))
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()
    dt = float(env._engine.get_timestep())
    character = env._get_char_id()
    samples = []
    ending = 'unfinished'
    with torch.no_grad():
        for step in range(400):
            action, _ = agent._decide_action(obs, info)
            obs, _, done, info = env.step(action)
            actual = float(torch.linalg.norm(env._engine.get_root_vel(character)[0, :2]).item())
            reference = float(torch.linalg.norm(env._ref_root_vel[0, :2]).item())
            samples.append((step * dt, actual, reference))
            flag = int(done[0].item())
            if flag != base_env.DoneFlags.NULL.value:
                ending = next((name for name in ('SUCC', 'FAIL', 'TIME')
                               if flag == getattr(base_env.DoneFlags, name).value), str(flag))
                break
    bins = []
    n_per_bin = max(1, round(bin_seconds / dt))
    for start in range(0, len(samples), n_per_bin):
        part = samples[start:start + n_per_bin]
        bins.append(dict(time_s=[round(part[0][0], 3), round(part[-1][0] + dt, 3)],
                         actual_mean_mps=round(sum(x[1] for x in part) / len(part), 3),
                         reference_mean_mps=round(sum(x[2] for x in part) / len(part), 3)))
    return dict(env_config=str(env_config), model_file=str(model_file), seed=seed,
                control_dt_s=dt, steps=len(samples), end=ending,
                actual_mean_mps=round(sum(x[1] for x in samples) / len(samples), 3),
                reference_mean_mps=round(sum(x[2] for x in samples) / len(samples), 3),
                speed_bins=bins)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', type=Path, required=True)
    parser.add_argument('--model-file', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--bin-seconds', type=float, default=0.5)
    args = parser.parse_args()
    if args.bin_seconds <= 0:
        parser.error('--bin-seconds must be positive')
    print('SPEED_PROFILE ' + json.dumps(measure(args.env_config, args.model_file,
                                                args.seed, args.bin_seconds)), flush=True)
