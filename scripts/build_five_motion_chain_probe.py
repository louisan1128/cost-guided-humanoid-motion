#!/usr/bin/env python3
"""Build a five-skill transition probe without training or blending motions.

Sequence: start -> walk -> left turn -> walk -> right turn -> walk -> stop.
The output is a kinematic reference for diagnosing the fixed PPO tracker.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation

from build_start_walk_probe import heading, load_motion, pose_distance, resample


MOTION_FILES = {
    "start": "mixamo_start_xyfix.pkl",
    "walk": "humanoid_walk_neck_fixed.pkl",
    "turn_left": "humanoid_turn_left_5s_8s.pkl",
    "turn_right": "humanoid_turn_right_5s_8s.pkl",
    "stop": "mixamo_stop_xyfix.pkl",
}
DEFAULT_SEQUENCE = ("start", "walk", "turn_left", "walk", "turn_right", "walk", "stop")


def one_walk_cycle(walk: np.ndarray, previous_pose: np.ndarray) -> tuple[np.ndarray, int]:
    # Exclude the duplicate loop endpoint while searching for a joint phase.
    unique = walk[:-1]
    scores = np.asarray([pose_distance(previous_pose, frame) for frame in unique])
    phase = int(np.argmin(scores))
    period = len(unique)
    step = np.arange(period + 1)
    frame_index = (phase + step) % period
    cycle_index = (phase + step) // period
    cycle = unique[frame_index].copy()
    cycle[:, :3] += cycle_index[:, None] * (walk[-1, :3] - walk[0, :3])
    return cycle, phase


def align_root(segment: np.ndarray, previous_pose: np.ndarray) -> np.ndarray:
    aligned = segment.copy()
    yaw_delta = heading(previous_pose[3:6]) - heading(segment[0, 3:6])
    yaw_rotation = Rotation.from_euler("z", yaw_delta)
    aligned[:, :3] = previous_pose[:3] + yaw_rotation.apply(segment[:, :3] - segment[0, :3])
    aligned[:, 3:6] = (yaw_rotation * Rotation.from_rotvec(segment[:, 3:6])).as_rotvec()
    return aligned


def build(source_dir: Path, output: Path, sequence: tuple[str, ...] = DEFAULT_SEQUENCE) -> dict:
    report_path = output.with_suffix(".json")
    if output.exists() or report_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output} or {report_path}")

    loaded = {}
    source_records = {}
    if len(sequence) < 2 or any(label not in MOTION_FILES for label in sequence):
        raise ValueError(f"Expected at least two known motions: {sequence}")
    for label in sequence:
        name = MOTION_FILES[label]
        if name in loaded:
            continue
        path = source_dir / name
        motion, frames = load_motion(path)
        loaded[name] = (motion, frames)
        source_records[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    fps = float(loaded["humanoid_walk_neck_fixed.pkl"][0]["fps"])
    if int(loaded["humanoid_walk_neck_fixed.pkl"][0]["loop_mode"]) != 1:
        raise ValueError("The walk source must be a looping motion")

    combined = None
    boundaries = []
    for label in sequence:
        name = MOTION_FILES[label]
        motion, raw = loaded[name]
        frames = resample(raw, float(motion["fps"]), fps) if float(motion["fps"]) != fps else raw.copy()
        phase = None
        if label == "walk" and combined is not None:
            frames, phase = one_walk_cycle(frames, combined[-1])

        if combined is None:
            combined = frames
            continue

        frames = align_root(frames, combined[-1])
        boundary_index = len(combined) - 1
        before = combined[-1]
        target_start = frames[0]
        after = frames[1]
        first_velocity = (combined[-1, :3] - combined[-2, :3]) * fps
        next_velocity = (after[:3] - before[:3]) * fps
        boundaries.append({
            "into": label,
            "reference_boundary_frame": boundary_index,
            "reference_boundary_time_s": boundary_index / fps,
            "walk_phase_frame": phase,
            "target_start_joint_pose_distance_rad": pose_distance(before, target_start),
            "one_frame_joint_pose_distance_rad": pose_distance(before, after),
            "target_start_root_rotation_rad": float(np.linalg.norm(
                (Rotation.from_rotvec(before[3:6]).inv() * Rotation.from_rotvec(target_start[3:6])).as_rotvec())),
            "one_frame_root_step_m": float(np.linalg.norm(after[:3] - before[:3])),
            "root_velocity_jump_mps": float(np.linalg.norm(next_velocity - first_velocity)),
        })
        # The previous endpoint appears once; the next source starts one frame later.
        combined = np.concatenate((combined, frames[1:]), axis=0)

    assert combined is not None
    if not np.isfinite(combined).all():
        raise ValueError("Combined reference contains non-finite values")

    report = {
        "purpose": "diagnostic five-skill PPO tracker transitions in one physical episode",
        "sequence": list(sequence),
        "source_files": source_records,
        "operation": "resample to 120 fps; choose nearest walk joint phase; align each new root position and heading; no blend or physics optimization",
        "fps": fps,
        "frames": len(combined),
        "duration_s": (len(combined) - 1) / fps,
        "boundaries": boundaries,
        "status": "kinematic reference only; physical PPO tracking untested",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"loop_mode": 0, "fps": fps,
                     "frames": combined.astype(np.float32).tolist()}, stream)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sequence", nargs="+", choices=tuple(MOTION_FILES), default=DEFAULT_SEQUENCE)
    args = parser.parse_args()
    print(json.dumps(build(args.source_dir, args.output, tuple(args.sequence)), indent=2))
