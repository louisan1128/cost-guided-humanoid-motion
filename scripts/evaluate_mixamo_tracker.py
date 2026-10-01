#!/usr/bin/env python3
"""Evaluate a trained MimicKit PPO tracker from the first motion frame.

The action comes from the loaded policy. The physical character is reset only
between episodes; FAIL/SUCC/TIME are reported separately.
"""
import argparse
import json
import os
from pathlib import Path
import sys

MIMICKIT = Path('/home/an/Desktop/MimicKit-main')
sys.path.insert(0, str(MIMICKIT / 'mimickit'))

import torch
import envs.base_env as base_env
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.util as mimic_util


def evaluate(env_config: Path, model_file: Path, episodes: int, seed: int) -> dict:
    env_config, model_file = env_config.resolve(), model_file.resolve()
    os.chdir(MIMICKIT)
    device = 'cuda:0'
    mp_util.init(0, 1, device, 6341)
    mimic_util.set_rand_seed(seed)
    env = env_builder.build_env(str(env_config), 'data/engines/isaac_lab_engine.yaml',
                                episodes, device, visualize=False)
    agent = agent_builder.build_agent('data/agents/deepmimic_humanoid_ppo_agent.yaml', env, device)
    agent.load(str(model_file))
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()
    finished = torch.zeros(episodes, dtype=torch.bool, device=device)
    flags = torch.zeros(episodes, dtype=torch.int32, device=device)
    lengths = torch.zeros(episodes, dtype=torch.int32, device=device)
    root_error_sum = torch.zeros(episodes, dtype=torch.float32, device=device)
    body_error_sum = torch.zeros(episodes, dtype=torch.float32, device=device)
    body_error_max = torch.zeros(episodes, dtype=torch.float32, device=device)
    char_id = env._get_char_id()
    with torch.no_grad():
        for _ in range(400):
            action, _ = agent._decide_action(obs, info)
            obs, reward, done, info = env.step(action)
            active = ~finished
            lengths[active] += 1
            root_error = torch.linalg.norm(env._engine.get_root_pos(char_id) - env._ref_root_pos, dim=-1)
            body_error = torch.linalg.norm(env._engine.get_body_pos(char_id) - env._ref_body_pos, dim=-1).mean(dim=-1)
            root_error_sum[active] += root_error[active]
            body_error_sum[active] += body_error[active]
            body_error_max[active] = torch.maximum(body_error_max[active], body_error[active])
            first_done = active & (done != base_env.DoneFlags.NULL.value)
            flags[first_done] = done[first_done]
            finished |= first_done
            if bool(torch.all(finished)):
                break
            reset_ids = torch.nonzero(done != base_env.DoneFlags.NULL.value).flatten()
            if len(reset_ids):
                obs, info = env.reset(reset_ids)
    return dict(model=str(model_file), motion=env_config.name, episodes=episodes, seed=seed,
                success=int((flags == base_env.DoneFlags.SUCC.value).sum().item()),
                fail=int((flags == base_env.DoneFlags.FAIL.value).sum().item()),
                timeout=int((flags == base_env.DoneFlags.TIME.value).sum().item()),
                unfinished=int((flags == base_env.DoneFlags.NULL.value).sum().item()),
                episode_steps_mean=float(lengths.float().mean().item()),
                episode_steps_min=int(lengths.min().item()),
                episode_steps_max=int(lengths.max().item()),
                root_position_error_mean_m=float((root_error_sum / lengths.clamp_min(1)).mean().item()),
                body_position_error_mean_m=float((body_error_sum / lengths.clamp_min(1)).mean().item()),
                body_position_error_max_m=float(body_error_max.max().item()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-config', type=Path, required=True)
    parser.add_argument('--model-file', type=Path, required=True)
    parser.add_argument('--episodes', type=int, default=32)
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.episodes <= 0:
        parser.error('--episodes must be positive')
    print('TRACKER_EVAL ' + json.dumps(evaluate(args.env_config, args.model_file,
                                              args.episodes, args.seed)), flush=True)
