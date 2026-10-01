#!/usr/bin/env python3
"""Insert a short, support-foot-locked bridge at one diagnostic motion join.

This makes a kinematic reference for testing. It does not claim that the
result is dynamically feasible or learned by the fixed PPO tracker.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import sys

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from build_contact_aligned_turn_probe import FOOT_NAMES, sole_positions
from build_start_walk_probe import BALL_ROTATIONS, HINGES, load_motion


def interpolate_pose(a: np.ndarray, b: np.ndarray, u: float) -> np.ndarray:
    w = u * u * (3.0 - 2.0 * u)
    pose = (1.0 - w) * a + w * b
    for block in (slice(3, 6), *BALL_ROTATIONS):
        key = Rotation.from_rotvec(np.stack((a[block], b[block])))
        pose[block] = Slerp([0.0, 1.0], key)([w]).as_rotvec()[0]
    for idx in HINGES:
        pose[idx] = (1.0 - w) * a[idx] + w * b[idx]
    return pose


def build(motion_path: Path, source_report: Path, output: Path,
          duration_s: float, mimickit_root: Path, support_name: str | None) -> dict:
    target_report = output.with_suffix('.json')
    if output.exists() or target_report.exists():
        raise FileExistsError(f'Refusing to overwrite {output} or {target_report}')
    info, frames = load_motion(motion_path)
    source = json.loads(source_report.read_text())
    boundary = source.get('boundary_frame')
    if boundary is None:
        boundary = source['boundaries'][0]['reference_boundary_frame']
    boundary = int(boundary)
    fps = float(info['fps'])
    bridge_steps = round(duration_s * fps)
    if not (2 <= bridge_steps and 2 <= boundary < len(frames) - 2):
        raise ValueError('Bridge duration/boundary is invalid')
    if support_name is None:
        support_name = source.get('support_foot')
    if support_name not in FOOT_NAMES:
        raise ValueError('Specify --support left_foot or right_foot')
    support = FOOT_NAMES.index(support_name)
    sys.path.insert(0, str(mimickit_root / 'mimickit'))
    from anim.mjcf_char_model import MJCFCharModel
    from util import torch_util
    model = MJCFCharModel('cpu')
    model.load(str(mimickit_root / 'data/assets/humanoid/humanoid.xml'))

    first = frames[boundary].copy()
    next_frame = frames[boundary + 1].copy()
    feet_before = sole_positions(frames[boundary:boundary + 2], model, torch_util)
    support_shift = feet_before[0, support] - feet_before[1, support]
    following = frames[boundary + 1:].copy()
    following[:, :3] += support_shift
    corrected_next = following[0]

    bridge = np.stack([interpolate_pose(first, corrected_next, j / bridge_steps)
                       for j in range(1, bridge_steps)], axis=0)
    # Translate the root of each intermediate pose to keep the planted sole fixed.
    bridge_feet = sole_positions(bridge, model, torch_util)
    root_corrections = feet_before[0, support] - bridge_feet[:, support]
    bridge[:, :3] += root_corrections
    combined = np.concatenate((frames[:boundary + 1], bridge, following), axis=0)
    if not np.isfinite(combined).all():
        raise ValueError('Non-finite generated reference')
    original_step = np.linalg.norm(feet_before[1] - feet_before[0], axis=1)
    new_feet = sole_positions(combined[boundary:boundary + bridge_steps + 1], model, torch_util)
    new_steps = np.linalg.norm(np.diff(new_feet, axis=0), axis=2)
    report = {
        'purpose': 'short support-foot-locked kinematic bridge at one motion join',
        'source_motion': str(motion_path.resolve()),
        'source_sha256': hashlib.sha256(motion_path.read_bytes()).hexdigest(),
        'source_report': str(source_report.resolve()),
        'support_foot': support_name,
        'method': 'rigidly translate following clip to match planted sole, interpolate root/joints with smoothstep and quaternion slerp, pin support sole by root translation; no IK or physics optimization',
        'bridge_duration_s': bridge_steps / fps,
        'bridge_frame_count': bridge_steps - 1,
        'following_clip_root_shift_m': support_shift.tolist(),
        'max_intermediate_root_correction_m': float(np.linalg.norm(root_corrections, axis=1).max()),
        'original_one_frame_sole_step_m': dict(zip(FOOT_NAMES, (float(v) for v in original_step))),
        'max_bridge_one_frame_sole_step_m': dict(zip(FOOT_NAMES, (float(v) for v in new_steps.max(axis=0)))),
        'fps': fps,
        'boundary_frame': boundary,
        'boundary_time_s': boundary / fps,
        'duration_s': (len(combined) - 1) / fps,
        'status': 'kinematic bridge; fixed PPO physical tracking not yet tested',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('wb') as stream:
        pickle.dump({'loop_mode': 0, 'fps': fps,
                     'frames': combined.astype(np.float32).tolist()}, stream)
    target_report.write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--source-report', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--duration', type=float, default=0.25)
    parser.add_argument('--support', choices=FOOT_NAMES)
    parser.add_argument('--mimickit-root', type=Path, default=Path('/home/an/Desktop/MimicKit-main'))
    args = parser.parse_args()
    print(json.dumps(build(args.motion, args.source_report, args.output,
                           args.duration, args.mimickit_root, args.support), indent=2))
