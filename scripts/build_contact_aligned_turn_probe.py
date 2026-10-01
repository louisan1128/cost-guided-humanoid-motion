#!/usr/bin/env python3
"""Pick a walk ending phase with the next clip's support foot, then stitch.

This is a kinematic diagnostic, not a physically validated transition policy.
The walk starts at its original frame zero so the baseline initial condition is
preserved; only its duration changes before the next reference starts.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from build_five_motion_chain_probe import align_root
from build_start_walk_probe import load_motion, pose_distance, resample


FOOT_NAMES = ("left_foot", "right_foot")
# humanoid.xml: box centre [0.045, 0, -0.0225], half-height 0.0275.
SOLE_LOCAL = (0.045, 0.0, -0.05)


def sole_positions(frames: np.ndarray, model, torch_util) -> np.ndarray:
    data = torch.as_tensor(frames, dtype=torch.float32)
    with torch.no_grad():
        body_pos, body_rot = model.forward_kinematics(
            data[:, :3], torch_util.exp_map_to_quat(data[:, 3:6]),
            model.dof_to_rot(data[:, 6:]))
        offset = torch.tensor(SOLE_LOCAL, dtype=torch.float32).expand(len(data), 3)
        feet = [body_pos[:, model.get_body_id(name)] +
                torch_util.quat_rotate(body_rot[:, model.get_body_id(name)], offset)
                for name in FOOT_NAMES]
    return torch.stack(feet, dim=1).numpy()


def build(walk_path: Path, turn_path: Path, output: Path, mimickit_root: Path,
          extra_walk_cycles: int = 0) -> dict:
    if extra_walk_cycles < 0:
        raise ValueError("extra_walk_cycles must be non-negative")
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    sys.path.insert(0, str(mimickit_root / "mimickit"))
    from anim.mjcf_char_model import MJCFCharModel
    from util import torch_util

    walk_info, walk = load_motion(walk_path)
    turn_info, turn = load_motion(turn_path)
    fps = float(walk_info["fps"])
    if int(walk_info["loop_mode"]) != 1:
        raise ValueError("Expected a looping walk source")
    if abs(float(turn_info["fps"]) - fps) > 1e-4:
        turn = resample(turn, float(turn_info["fps"]), fps)

    model = MJCFCharModel("cpu")
    model.load(str(mimickit_root / "data/assets/humanoid/humanoid.xml"))
    walk_feet = sole_positions(walk, model, torch_util)
    turn_feet = sole_positions(turn[:13], model, torch_util)
    turn_height = turn_feet[0, :, 2]
    turn_speed = np.linalg.norm(np.diff(turn_feet[:, :, :2], axis=0) * fps, axis=2).mean(axis=0)
    support = int(np.argmin(turn_height))
    if turn_speed[support] >= 0.2:
        raise ValueError(f"Turn start has no nearly stationary low foot: {turn_speed}")

    period = len(walk) - 1
    cycle_delta = walk[-1, :3] - walk[0, :3]
    candidates = []
    for phase in range(period):
        ending = walk[phase].copy()
        ending[:3] += cycle_delta
        ending_feet = walk_feet[phase] + cycle_delta
        next_phase = (phase + 1) % period
        next_feet = walk_feet[next_phase].copy()
        if next_phase == 0:
            next_feet += cycle_delta
        foot_speed = float(np.linalg.norm((next_feet[support, :2] - walk_feet[phase, support, :2]) * fps))
        other = 1 - support
        if ending_feet[support, 2] > ending_feet[other, 2] - 0.02 or foot_speed >= 0.1:
            continue
        aligned_turn = align_root(turn[:2], ending)
        incoming_feet = sole_positions(aligned_turn[:1], model, torch_util)[0]
        gaps = np.linalg.norm(incoming_feet - ending_feet, axis=1)
        candidates.append((float(np.max(gaps)), phase, foot_speed, gaps))
    if not candidates:
        raise ValueError("No walk phase has the turn's support foot planted")

    worst_gap, phase, support_speed, gaps = min(candidates, key=lambda item: (item[0], item[1]))
    parts = [walk]
    for cycle_number in range(1, extra_walk_cycles + 1):
        repeated = walk[1:].copy()
        repeated[:, :3] += cycle_number * cycle_delta
        parts.append(repeated)
    tail = walk[1:phase + 1].copy()
    tail[:, :3] += (extra_walk_cycles + 1) * cycle_delta
    parts.append(tail)
    walk_segment = np.concatenate(parts, axis=0)
    aligned_turn = align_root(turn, walk_segment[-1])
    boundary = len(walk_segment) - 1
    combined = np.concatenate((walk_segment, aligned_turn[1:]), axis=0)
    joined_feet = sole_positions(combined[boundary:boundary + 2], model, torch_util)
    one_frame_gap = np.linalg.norm(joined_feet[1] - joined_feet[0], axis=1)
    if not np.isfinite(combined).all():
        raise ValueError("Combined motion contains non-finite frames")

    report = {
        "purpose": "contact-aware walk-to-target reference diagnostic for fixed PPO tracker",
        "walk_source": str(walk_path.resolve()),
        "walk_sha256": hashlib.sha256(walk_path.read_bytes()).hexdigest(),
        "turn_source": str(turn_path.resolve()),
        "turn_sha256": hashlib.sha256(turn_path.read_bytes()).hexdigest(),
        "method": "preserve walk frame-zero start; add requested full cycles and selected phase; require same low/stationary support foot; minimize maximum 3D sole gap; align target root position/heading; no blend",
        "support_foot": FOOT_NAMES[support],
        "turn_start_support_xy_speed_mps": float(turn_speed[support]),
        "walk_end_support_xy_speed_mps": support_speed,
        "selected_walk_phase_frame": phase,
        "extra_walk_cycles": extra_walk_cycles,
        "candidate_phases": len(candidates),
        "fps": fps,
        "boundary_frame": boundary,
        "boundary_time_s": boundary / fps,
        "frames": len(combined),
        "duration_s": (len(combined) - 1) / fps,
        "target_start_sole_gap_m": dict(zip(FOOT_NAMES, (float(x) for x in gaps))),
        "target_start_worst_sole_gap_m": worst_gap,
        "one_frame_sole_gap_m": dict(zip(FOOT_NAMES, (float(x) for x in one_frame_gap))),
        "joint_pose_distance_rad": pose_distance(walk_segment[-1], aligned_turn[0]),
        "status": "kinematic diagnostic; physical PPO tracking not yet tested",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"loop_mode": 0, "fps": fps,
                     "frames": combined.astype(np.float32).tolist()}, stream)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walk", type=Path, required=True)
    parser.add_argument("--turn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mimickit-root", type=Path, default=Path("/home/an/Desktop/MimicKit-main"))
    parser.add_argument("--extra-walk-cycles", type=int, default=0)
    args = parser.parse_args()
    print(json.dumps(build(args.walk, args.turn, args.output, args.mimickit_root,
                           args.extra_walk_cycles), indent=2))
