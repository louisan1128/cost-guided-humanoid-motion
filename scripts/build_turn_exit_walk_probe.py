#!/usr/bin/env python3
"""Join a natural turn exit to a walk cycle at a compatible planted foot.

This is a kinematic probe. It does not blend or physically optimize the join.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from build_contact_aligned_turn_probe import FOOT_NAMES, sole_positions
from build_five_motion_chain_probe import align_root
from build_start_walk_probe import heading, load_motion, pose_distance


def build(source: Path, walk_path: Path, output: Path, mimickit_root: Path,
          start_s: float = 5.0, search_start_s: float = 8.0,
          search_end_s: float = 9.2, max_yaw_rate: float = 0.35) -> dict:
    report_path = output.with_suffix('.json')
    if output.exists() or report_path.exists():
        raise FileExistsError(f'Refusing to overwrite {output} or {report_path}')
    sys.path.insert(0, str(mimickit_root / 'mimickit'))
    from anim.mjcf_char_model import MJCFCharModel
    from util import torch_util

    turn_info, turn = load_motion(source)
    walk_info, walk = load_motion(walk_path)
    fps = float(turn_info['fps'])
    if abs(float(walk_info['fps']) - fps) > 1e-4 or int(walk_info['loop_mode']) != 1:
        raise ValueError('Turn and looping walk must share fps')
    first = round(start_s * fps)
    lo = round(search_start_s * fps)
    hi = round(search_end_s * fps)
    if not (0 <= first < lo <= hi + 6 < len(turn)):
        raise ValueError('Invalid crop/search range')

    model = MJCFCharModel('cpu')
    model.load(str(mimickit_root / 'data/assets/humanoid/humanoid.xml'))
    turn_feet = sole_positions(turn[first:hi + 7], model, torch_util)
    walk_feet = sole_positions(walk, model, torch_util)
    period = len(walk) - 1
    walk_speed = np.linalg.norm(np.diff(walk_feet[:period + 1, :, :2], axis=0) * fps, axis=2)
    walk_heading = np.array([heading(frame[3:6]) for frame in walk[:period]])
    candidates = []
    # Each candidate has a 30 Hz exit time; the reference itself stays at 120 Hz.
    for end in range(lo, hi + 1, 4):
        local = end - first
        endpoint = turn[end]
        speed = float(np.linalg.norm((turn[end + 6, :2] - turn[end - 6, :2]) * fps / 12))
        yaw_change = Rotation.from_rotvec(turn[end - 6, 3:6]).inv() * Rotation.from_rotvec(turn[end + 6, 3:6])
        yaw_rate = abs(float(heading(yaw_change.as_rotvec()) * fps / 12))
        if not (0.5 <= speed <= 1.6 and yaw_rate <= max_yaw_rate):
            continue
        feet_end = turn_feet[local]
        feet_speed = np.linalg.norm((turn_feet[local + 1, :, :2] - turn_feet[local - 1, :, :2]) * fps / 2, axis=1)
        support = int(np.argmin(feet_end[:, 2]))
        other = 1 - support
        if feet_end[support, 2] > feet_end[other, 2] - 0.02 or feet_speed[support] >= 0.12:
            continue
        for phase in range(period):
            if walk_feet[phase, support, 2] > walk_feet[phase, other, 2] - 0.02:
                continue
            if walk_speed[phase, support] >= 0.12:
                continue
            yaw_delta = heading(endpoint[3:6]) - walk_heading[phase]
            rotation = Rotation.from_euler('z', yaw_delta)
            aligned_feet = endpoint[:3] + rotation.apply(walk_feet[phase] - walk[phase, :3])
            gaps = np.linalg.norm(aligned_feet - feet_end, axis=1)
            worst = float(np.max(gaps))
            if worst >= 0.12:
                continue
            pose = pose_distance(endpoint, walk[phase])
            score = worst + 0.02 * pose + 0.02 * abs(speed - 1.0) + 0.03 * yaw_rate
            candidates.append((score, end, phase, support, worst, pose, speed, yaw_rate, gaps))
    if not candidates:
        raise ValueError('No exit with matching planted foot, yaw rate and sole positions')
    score, end, phase, support, gap, pose, speed, yaw_rate, gaps = min(candidates, key=lambda row: (row[0], row[1], row[2]))
    turn_segment = turn[first:end + 1].copy()
    # Match the +X root normalization used by the trained 5–8 s turn clips.
    initial_yaw = heading(turn_segment[0, 3:6])
    world_rotation = Rotation.from_euler("z", -initial_yaw)
    origin_xy = turn_segment[0, :2].copy()
    xy_offset = turn_segment[:, :2] - origin_xy
    turn_segment[:, :2] = world_rotation.apply(
        np.column_stack((xy_offset, np.zeros(len(turn_segment))))
    )[:, :2]
    turn_segment[:, 3:6] = (
        world_rotation * Rotation.from_rotvec(turn_segment[:, 3:6])
    ).as_rotvec()
    indexes = (phase + np.arange(period + 1)) % period
    cycles = (phase + np.arange(period + 1)) // period
    cycle = walk[indexes].copy()
    cycle[:, :3] += cycles[:, None] * (walk[-1, :3] - walk[0, :3])
    aligned_walk = align_root(cycle, turn_segment[-1])
    combined = np.concatenate((turn_segment, aligned_walk[1:]), axis=0)
    boundary = len(turn_segment) - 1
    one_frame_feet = sole_positions(combined[boundary:boundary + 2], model, torch_util)
    one_frame_gap = np.linalg.norm(one_frame_feet[1] - one_frame_feet[0], axis=1)
    if not np.isfinite(combined).all():
        raise ValueError('Non-finite joined motion')
    report = {
        'purpose': 'turn-to-walk support-foot-aligned diagnostic',
        'turn_source': str(source.resolve()),
        'turn_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'walk_source': str(walk_path.resolve()),
        'walk_sha256': hashlib.sha256(walk_path.read_bytes()).hexdigest(),
        'method': 'rebase turn start to +X and root XY zero; select low-yaw-rate exit and same planted walk foot; rigid root alignment; no blend or physics optimization',
        'turn_start_s': start_s,
        'turn_exit_source_s': end / fps,
        'selected_walk_phase_frame': phase,
        'support_foot': FOOT_NAMES[support],
        'candidate_count': len(candidates),
        'score': score,
        'turn_exit_root_xy_speed_mps': speed,
        'turn_exit_abs_yaw_rate_radps': yaw_rate,
        'max_yaw_rate_radps': max_yaw_rate,
        'target_start_worst_sole_gap_m': gap,
        'target_start_sole_gap_m': dict(zip(FOOT_NAMES, (float(v) for v in gaps))),
        'one_frame_sole_gap_m': dict(zip(FOOT_NAMES, (float(v) for v in one_frame_gap))),
        'joint_pose_distance_rad': pose,
        'fps': fps,
        'boundary_frame': boundary,
        'boundary_time_s': boundary / fps,
        'duration_s': (len(combined) - 1) / fps,
        'status': 'kinematic probe; physical PPO tracking not yet tested',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        pickle.dump({'loop_mode': 0, 'fps': fps, 'frames': combined.astype(np.float32).tolist()}, stream)
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, type=Path)
    parser.add_argument('--walk', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mimickit-root', type=Path, default=Path('/home/an/Desktop/MimicKit-main'))
    parser.add_argument('--start', type=float, default=5.0)
    parser.add_argument('--search-start', type=float, default=8.0)
    parser.add_argument('--search-end', type=float, default=9.2)
    parser.add_argument('--max-yaw-rate', type=float, default=0.35)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.walk, args.output, args.mimickit_root,
                           args.start, args.search_start, args.search_end, args.max_yaw_rate), indent=2))
