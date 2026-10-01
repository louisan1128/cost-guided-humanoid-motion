#!/usr/bin/env python3
"""Build a diagnostic start→walk reference for the existing MimicKit tracker.

This joins two source clips into one non-looping reference. It does not claim
that the join is physically feasible or that PPO has learned transitions.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


ROOT_ROT = slice(3, 6)
BALL_ROTATIONS = (slice(6, 9), slice(9, 12), slice(12, 15),
                  slice(16, 19), slice(20, 23), slice(24, 27),
                  slice(27, 30), slice(31, 34))
HINGES = (15, 19, 23, 30)


def load_motion(path: Path) -> tuple[dict, np.ndarray]:
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    frames = np.asarray(motion["frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 34 or not np.isfinite(frames).all():
        raise ValueError(f"Expected finite 34-D generic humanoid frames: {path}")
    if float(motion["fps"]) <= 0 or len(frames) < 3:
        raise ValueError(f"Invalid fps or frame count: {path}")
    return motion, frames


def resample(frames: np.ndarray, src_fps: float, dst_fps: float) -> np.ndarray:
    duration = (len(frames) - 1) / src_fps
    intervals = round(duration * dst_fps)
    times = np.linspace(0.0, duration, intervals + 1)
    src_times = np.arange(len(frames)) / src_fps
    result = np.column_stack([np.interp(times, src_times, frames[:, i])
                              for i in range(frames.shape[1])])
    for block in (ROOT_ROT, *BALL_ROTATIONS):
        result[:, block] = Slerp(src_times, Rotation.from_rotvec(frames[:, block]))(times).as_rotvec()
    return result


def pose_distance(a: np.ndarray, b: np.ndarray) -> float:
    value = sum(np.linalg.norm((Rotation.from_rotvec(a[s]).inv()
                                * Rotation.from_rotvec(b[s])).as_rotvec()) ** 2
                for s in BALL_ROTATIONS)
    value += sum(float(a[i] - b[i]) ** 2 for i in HINGES)
    return float(np.sqrt(value))


def heading(rotation: np.ndarray) -> float:
    forward = Rotation.from_rotvec(rotation).apply([1.0, 0.0, 0.0])
    return float(np.arctan2(forward[1], forward[0]))


def build(start_path: Path, walk_path: Path, output: Path, cycles: int) -> dict:
    if output.exists() or output.with_suffix(".json").exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if cycles < 1:
        raise ValueError("cycles must be at least 1")

    start_motion, start_src = load_motion(start_path)
    walk_motion, walk = load_motion(walk_path)
    if int(walk_motion["loop_mode"]) != 1:
        raise ValueError("Walk source must be a looping motion")
    fps = float(walk_motion["fps"])
    start = resample(start_src, float(start_motion["fps"]), fps)
    unique_walk = walk[:-1]

    # Select the walk phase closest to the final start *joint pose*.
    # The search uses no future simulator state and does not synthesize a bridge.
    distances = np.asarray([pose_distance(start[-1], frame) for frame in unique_walk])
    phase = int(np.argmin(distances))
    period = len(unique_walk)
    step = np.arange(cycles * period + 1)
    source_index = (phase + step) % period
    cycle_index = (phase + step) // period
    walk_sequence = unique_walk[source_index].copy()
    root_cycle_delta = walk[-1, :3] - walk[0, :3]
    walk_sequence[:, :3] += cycle_index[:, None] * root_cycle_delta

    yaw_delta = heading(start[-1, ROOT_ROT]) - heading(walk_sequence[0, ROOT_ROT])
    yaw_rotation = Rotation.from_euler("z", yaw_delta)
    relative_pos = walk_sequence[:, :3] - walk_sequence[0, :3]
    walk_sequence[:, :3] = start[-1, :3] + yaw_rotation.apply(relative_pos)
    walk_sequence[:, ROOT_ROT] = (yaw_rotation * Rotation.from_rotvec(walk_sequence[:, ROOT_ROT])).as_rotvec()

    # Keep the start endpoint once. The next sample is the first walk step.
    boundary = len(start) - 1
    combined = np.concatenate((start, walk_sequence[1:]), axis=0)
    if not np.isfinite(combined).all():
        raise ValueError("Combined reference contains non-finite values")
    pos_step = float(np.linalg.norm(combined[boundary + 1, :3] - combined[boundary, :3]))
    rot_step = float(np.linalg.norm((Rotation.from_rotvec(combined[boundary, ROOT_ROT]).inv()
                                     * Rotation.from_rotvec(combined[boundary + 1, ROOT_ROT])).as_rotvec()))
    joint_step = pose_distance(combined[boundary], combined[boundary + 1])
    start_velocity = (combined[boundary, :3] - combined[boundary - 1, :3]) * fps
    walk_velocity = (combined[boundary + 1, :3] - combined[boundary, :3]) * fps

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"loop_mode": 0, "fps": fps,
                     "frames": combined.astype(np.float32).tolist()}, stream)
    report = {
        "purpose": "diagnostic PPO tracking of start→walk in one physical episode",
        "start_source": str(start_path.resolve()),
        "start_sha256": hashlib.sha256(start_path.read_bytes()).hexdigest(),
        "walk_source": str(walk_path.resolve()),
        "walk_sha256": hashlib.sha256(walk_path.read_bytes()).hexdigest(),
        "operation": "resample start to walk fps; choose nearest walk joint pose; align walk root XYZ and heading; concatenate without blend",
        "selected_walk_frame": phase,
        "selected_walk_phase": phase / period,
        "walk_cycles": cycles,
        "fps": fps,
        "frames": len(combined),
        "boundary_frame": boundary,
        "boundary_time_s": boundary / fps,
        "duration_s": (len(combined) - 1) / fps,
        "joint_pose_distance_at_selected_phase_rad": float(distances[phase]),
        "boundary_one_frame_root_step_m": pos_step,
        "boundary_one_frame_root_rotation_rad": rot_step,
        "boundary_one_frame_joint_pose_distance_rad": joint_step,
        "start_root_velocity_mps": start_velocity.tolist(),
        "walk_root_velocity_mps": walk_velocity.tolist(),
        "root_velocity_jump_mps": float(np.linalg.norm(walk_velocity - start_velocity)),
        "status": "kinematic reference only; physical PPO tracking untested",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=Path, required=True)
    parser.add_argument("--walk", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(build(args.start, args.walk, args.output, args.cycles), indent=2))
