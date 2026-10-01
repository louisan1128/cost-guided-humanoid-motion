#!/usr/bin/env python3
"""Short Isaac Lab physical PD tracking diagnostic, without PPO training.

The humanoid is initialized to the first reference frame once. Afterwards only
joint position targets are sent; no root/body pose is forced during rollout.
This is a controller-specific diagnostic, not proof of physical feasibility.
"""
import argparse
import json
import os
from pathlib import Path
import sys

MIMICKIT = Path('/home/an/Desktop/MimicKit-main')
sys.path.insert(0, str(MIMICKIT / 'mimickit'))

import torch
import yaml
from scipy.spatial.transform import Rotation
from envs.deepmimic_env import DeepMimicEnv


def run(motion_file: Path) -> dict:
    motion_file = motion_file.resolve()
    os.chdir(MIMICKIT)
    with (MIMICKIT / 'data/envs/deepmimic_humanoid_env.yaml').open() as f:
        env_cfg = yaml.safe_load(f)
    with (MIMICKIT / 'data/engines/isaac_lab_engine.yaml').open() as f:
        engine_cfg = yaml.safe_load(f)
    env_cfg.update(motion_file=str(motion_file.resolve()), rand_reset=False,
                   visualize_ref_char=False, log_tracking_error=False,
                   enable_early_termination=True)
    device = 'cuda:0'
    env = DeepMimicEnv(env_cfg, engine_cfg, num_envs=1, device=device,
                       visualize=False, record_video=False)
    env.reset()
    char_id = env._get_char_id()
    dt = env._engine.get_timestep()
    length = float(env._motion_lib.get_motion_length(env._motion_ids)[0].item())
    root_errors, root_heights, dof_errors = [], [], []
    done_value = 0
    failure_detail = {}
    for step in range(round(length / dt) + 2):
        target_time = min((step + 1) * dt, length)
        _, _, _, _, joint_rot, _ = env._motion_lib.calc_motion_frame(
            env._motion_ids, torch.tensor([target_time], device=device))
        cmd = env._motion_lib.joint_rot_to_dof(joint_rot)
        _, _, done, _ = env.step(cmd)
        root = env._engine.get_root_pos(char_id)[0]
        ref = env._ref_root_pos[0]
        dof = env._engine.get_dof_pos(char_id)[0]
        root_errors.append(float(torch.linalg.norm(root - ref).item()))
        root_heights.append(float(root[2].item()))
        dof_errors.append(float(torch.mean(torch.abs(dof - cmd[0])).item()))
        done_value = int(done[0].item())
        if done_value:
            body = env._engine.get_body_pos(char_id)[0]
            ref_body = env._ref_body_pos[0]
            body_error = torch.linalg.norm(body - ref_body, dim=-1)
            forces = env._engine.get_ground_contact_forces(char_id)[0]
            allowed = set(env._contact_body_ids.tolist())
            names = env._kin_char_model.get_body_names()
            unexpected = [names[i] for i in range(len(names))
                          if i not in allowed and torch.any(torch.abs(forces[i]) > 0.1)]
            root_quat = env._engine.get_root_rot(char_id)[0].cpu().numpy()
            root_up = Rotation.from_quat(root_quat).as_matrix()[:, 2]
            failure_detail = dict(max_body_error_m=float(body_error.max().item()),
                                  max_body_error_name=names[int(body_error.argmax().item())],
                                  unexpected_ground_contacts=unexpected,
                                  last_root_height_m=root_heights[-1],
                                  root_tilt_deg=float(torch.rad2deg(torch.arccos(
                                      torch.tensor(max(-1.0, min(1.0, float(root_up[2])))))).item()))
            break
    return dict(motion=motion_file.name, expected_duration_s=length,
                simulated_duration_s=round(len(root_heights) * dt, 3),
                termination_flag=done_value,
                min_root_height_m=min(root_heights),
                root_error_mean_m=sum(root_errors)/len(root_errors),
                root_error_max_m=max(root_errors),
                joint_target_error_mean_rad=sum(dof_errors)/len(dof_errors),
                controller='MimicKit Isaac Lab position target PD; no learned policy',
                failure_detail=failure_detail)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion_file', type=Path)
    args = parser.parse_args()
    print('PD_DIAGNOSTIC_RESULT ' + json.dumps(run(args.motion_file)), flush=True)
