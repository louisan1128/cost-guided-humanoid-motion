#!/usr/bin/env python3
"""Record real PPO foot contacts and tracking error around one reference join."""

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_mixamo_tracker import MIMICKIT
import envs.base_env as base_env
import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.util as mimic_util
import os
import torch
import yaml
from scipy.spatial.transform import Rotation


FOOT_NAMES = ("left_foot", "right_foot")


def mean(rows: list[dict], key: str) -> float | None:
    return sum(row[key] for row in rows) / len(rows) if rows else None


def trace(env_config: Path, model_file: Path, reference_report: Path,
          output_prefix: Path, seed: int) -> dict:
    env_config = env_config.resolve()
    model_file = model_file.resolve()
    reference_report = reference_report.resolve()
    output_prefix = output_prefix.resolve()
    csv_path = output_prefix.with_suffix(".csv")
    json_path = output_prefix.with_suffix(".json")
    if csv_path.exists() or json_path.exists():
        raise FileExistsError(f"Refusing to overwrite {csv_path} or {json_path}")
    report = json.loads(reference_report.read_text())
    boundary = float(report.get("boundary_time_s", report.get("boundaries", [{}])[0].get("reference_boundary_time_s")))
    cfg = yaml.safe_load(env_config.read_text())
    if Path(cfg["motion_file"]).resolve() != reference_report.with_suffix(".pkl"):
        raise ValueError("Environment motion_file and reference report do not match")
    if cfg["rand_reset"]:
        raise ValueError("Comparison requires a fixed first-frame reset")

    os.chdir(MIMICKIT)
    device = "cuda:0"
    mp_util.init(0, 1, device, 6341)
    mimic_util.set_rand_seed(seed)
    env = env_builder.build_env(str(env_config), "data/engines/isaac_lab_engine.yaml",
                                1, device, visualize=False)
    agent = agent_builder.build_agent("data/agents/deepmimic_humanoid_ppo_agent.yaml", env, device)
    agent.load(str(model_file))
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    obs, info = env.reset()
    dt = float(env._engine.get_timestep())
    char_id = env._get_char_id()
    names = env._kin_char_model.get_body_names()
    foot_ids = [names.index(name) for name in FOOT_NAMES]
    rows = []
    ending = "unfinished"

    with torch.no_grad():
        for step in range(int(cfg["episode_length"] / dt) + 4):
            action, _ = agent._decide_action(obs, info)
            obs, reward, done, info = env.step(action)
            actual_pos = env._engine.get_body_pos(char_id)[0]
            actual_vel = env._engine.get_body_vel(char_id)[0]
            ref_pos = env._ref_body_pos[0]
            forces = env._engine.get_ground_contact_forces(char_id)[0]
            actual_dof = env._engine.get_dof_pos(char_id)[0]
            ref_dof = env._kin_char_model.rot_to_dof(env._ref_joint_rot)[0]
            root_pos = env._engine.get_root_pos(char_id)[0]
            root_error = torch.linalg.norm(root_pos - env._ref_root_pos[0])
            root_quat = env._engine.get_root_rot(char_id)[0].cpu().numpy()
            root_up_z = float(Rotation.from_quat(root_quat).as_matrix()[2, 2])
            allowed_contacts = set(env._contact_body_ids.tolist())
            unexpected_contacts = sum(
                int(torch.linalg.norm(forces[i]).item() > 0.1)
                for i in range(len(names)) if i not in allowed_contacts)
            flag = int(done[0].item())
            row = {
                "step": step + 1,
                "time_s": round(float(env._time_buf[0].item()), 6),
                "reward": float(reward[0].item()),
                "root_pos_error_m": float(root_error.item()),
                "actual_root_height_m": float(root_pos[2].item()),
                "actual_root_xy_speed_mps": float(torch.linalg.norm(env._engine.get_root_vel(char_id)[0, :2]).item()),
                "actual_root_yaw_rate_radps": float(env._engine.get_root_ang_vel(char_id)[0, 2].item()),
                "actual_root_tilt_deg": math.degrees(math.acos(max(-1.0, min(1.0, root_up_z)))),
                "unexpected_contact_body_count": unexpected_contacts,
                "mean_abs_joint_error_rad": float(torch.mean(torch.abs(actual_dof - ref_dof)).item()),
                "done_flag": flag,
            }
            for foot, body_id in zip(FOOT_NAMES, foot_ids):
                row[f"{foot}_body_pos_error_m"] = float(torch.linalg.norm(actual_pos[body_id] - ref_pos[body_id]).item())
                row[f"{foot}_actual_xy_speed_mps"] = float(torch.linalg.norm(actual_vel[body_id, :2]).item())
                row[f"{foot}_ground_force_norm"] = float(torch.linalg.norm(forces[body_id]).item())
            rows.append(row)
            if flag != base_env.DoneFlags.NULL.value:
                ending = next((name for name in ("SUCC", "FAIL", "TIME")
                               if flag == getattr(base_env.DoneFlags, name).value), str(flag))
                break

    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    pre = [row for row in rows if boundary - 0.25 <= row["time_s"] < boundary]
    post = [row for row in rows if boundary <= row["time_s"] < boundary + 0.25]
    summary = {
        "env_config": str(env_config),
        "model_file": str(model_file),
        "reference_report": str(reference_report),
        "seed": seed,
        "control_dt_s": dt,
        "boundary_time_s": boundary,
        "episode_end_s": rows[-1]["time_s"],
        "survival_after_boundary_s": max(0.0, rows[-1]["time_s"] - boundary),
        "termination": ending,
        "pre_samples": len(pre),
        "post_samples": len(post),
        "pre_root_error_m": mean(pre, "root_pos_error_m"),
        "post_root_error_m": mean(post, "root_pos_error_m"),
        "pre_joint_error_rad": mean(pre, "mean_abs_joint_error_rad"),
        "post_joint_error_rad": mean(post, "mean_abs_joint_error_rad"),
        "last_root_height_m": rows[-1]["actual_root_height_m"],
        "last_root_xy_speed_mps": rows[-1]["actual_root_xy_speed_mps"],
        "last_root_yaw_rate_radps": rows[-1]["actual_root_yaw_rate_radps"],
        "last_root_tilt_deg": rows[-1]["actual_root_tilt_deg"],
        "last_unexpected_contact_body_count": rows[-1]["unexpected_contact_body_count"],
        "feet": {
            foot: {
                "pre_body_pos_error_m": mean(pre, f"{foot}_body_pos_error_m"),
                "post_body_pos_error_m": mean(post, f"{foot}_body_pos_error_m"),
                "pre_ground_force_norm": mean(pre, f"{foot}_ground_force_norm"),
                "post_ground_force_norm": mean(post, f"{foot}_ground_force_norm"),
                "pre_actual_xy_speed_mps": mean(pre, f"{foot}_actual_xy_speed_mps"),
                "post_actual_xy_speed_mps": mean(post, f"{foot}_actual_xy_speed_mps"),
            } for foot in FOOT_NAMES
        },
        "trace_csv": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--model-file", type=Path, required=True)
    parser.add_argument("--reference-report", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    print("TRANSITION_TRACE " + json.dumps(trace(args.env_config, args.model_file,
                                                args.reference_report, args.output_prefix,
                                                args.seed)), flush=True)
