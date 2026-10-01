#!/usr/bin/env python3
"""Join selected PPO-tracked transition references into one diagnostic chain.

This is a kinematic construction. The physical tracker must still be tested.
"""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation
import yaml

from build_start_walk_probe import BALL_ROTATIONS, heading, load_motion, pose_distance

ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "start_walk": "humanoid_start_to_walk_probe_v1",
    "walk_left": "inertial_walk_left_500_v1",
    "left_walk": "inertial_left_walk_300_v1_extended",
    "walk_right": "inertial_walk_right_500_v1",
    "right_walk": "inertial_right_walk_300_v1_extended",
    "walk_stop": "inertial_walk_stop_500_v1_hold5s_probe",
}


def boundary_frame(report: dict) -> int:
    if "boundary_frame" in report:
        return int(report["boundary_frame"])
    return int(report["boundaries"][0]["reference_boundary_frame"])


def choose_overlap(previous: np.ndarray, incoming: np.ndarray, incoming_boundary: int,
                   fps: float) -> tuple[int, int, dict]:
    # Match only the end of the previous clip against the part of the next
    # clip before its own transition. This keeps each skill transition intact.
    prev_candidates = range(max(1, len(previous) - round(1.5 * fps) - 1),
                            len(previous) - 2, 4)
    next_candidates = range(1, incoming_boundary - 8, 4)
    if not next_candidates:
        raise ValueError("Next clip has no pre-transition overlap window")
    best = None
    for i in prev_candidates:
        prev_vel = (previous[i + 1, 6:] - previous[i - 1, 6:]) * (fps / 2)
        for j in next_candidates:
            next_vel = (incoming[j + 1, 6:] - incoming[j - 1, 6:]) * (fps / 2)
            pose = float(np.linalg.norm(previous[i, 6:] - incoming[j, 6:]))
            vel = float(np.linalg.norm(prev_vel - next_vel))
            height = float(abs(previous[i, 2] - incoming[j, 2]))
            score = pose + 0.04 * vel + 2.0 * height
            if best is None or score < best[0]:
                best = (score, i, j, pose, vel, height)
    assert best is not None
    score, i, j, pose, vel, height = best
    return i, j, {"match_score": score, "joint_pose_gap_rad": pose,
                  "joint_velocity_gap_radps": vel, "root_height_gap_m": height}


def align_incoming(frames: np.ndarray, previous_pose: np.ndarray) -> np.ndarray:
    aligned = frames.copy()
    yaw_delta = heading(previous_pose[3:6]) - heading(frames[0, 3:6])
    rotation = Rotation.from_euler("z", yaw_delta)
    displacement = rotation.apply(frames[:, :3] - frames[0, :3])
    aligned[:, :2] = previous_pose[:2] + displacement[:, :2]
    # Keep the incoming root height. Inertialization handles the seam's
    # small height difference without shifting the entire clip off ground.
    aligned[:, 3:6] = (rotation * Rotation.from_rotvec(frames[:, 3:6])).as_rotvec()
    return aligned


def inertialize_join(frames: np.ndarray, boundary: int, n: int, fps: float) -> np.ndarray:
    if n < 3 or boundary < 2 or boundary + n + 1 >= len(frames):
        raise ValueError("Not enough frames for join correction")
    dt = 1.0 / fps
    duration = n / fps
    result = frames.copy()
    incoming_zero = 2 * frames[boundary + 1] - frames[boundary + 2]
    delta = frames[boundary] - incoming_zero
    derivative = ((frames[boundary] - frames[boundary - 1]) -
                  (frames[boundary + 2] - frames[boundary + 1])) / dt
    u = np.arange(1, n + 1) / n
    h0 = 1 - 10 * u**3 + 15 * u**4 - 6 * u**5
    h1 = u - 6 * u**3 + 8 * u**4 - 3 * u**5
    result[boundary + 1:boundary + n + 1] += h0[:, None] * delta + duration * h1[:, None] * derivative
    for block in (slice(3, 6), *BALL_ROTATIONS):
        old0 = Rotation.from_rotvec(frames[boundary, block])
        oldm = Rotation.from_rotvec(frames[boundary - 1, block])
        new1 = Rotation.from_rotvec(frames[boundary + 1, block])
        new2 = Rotation.from_rotvec(frames[boundary + 2, block])
        step = new2 * new1.inv()
        new0 = step.inv() * new1
        offset0 = (old0 * new0.inv()).as_rotvec()
        newm = step.inv() * new0
        offsetm = (oldm * newm.inv()).as_rotvec()
        offsetdot = (offset0 - offsetm) / dt
        correction = h0[:, None] * offset0 + duration * h1[:, None] * offsetdot
        result[boundary + 1:boundary + n + 1, block] = (
            Rotation.from_rotvec(correction) *
            Rotation.from_rotvec(frames[boundary + 1:boundary + n + 1, block])
        ).as_rotvec()
    return result


def build(turn: str, output: Path, env_output: Path, correction_s: float = 0.3) -> dict:
    if turn not in ("left", "right"):
        raise ValueError(turn)
    if correction_s <= 0:
        raise ValueError("Correction duration must be positive")
    for path in (output, output.with_suffix(".json"), env_output):
        if path.exists():
            raise FileExistsError(path)
    tags = ("start_walk", f"walk_{turn}", f"{turn}_walk", "walk_stop")
    loaded = []
    for tag in tags:
        source = ROOT / "data/reference_motions" / (SOURCES[tag] + ".pkl")
        motion, frames = load_motion(source)
        report = json.loads(source.with_suffix(".json").read_text())
        loaded.append((tag, source, motion, frames, report))
    fps = float(loaded[0][2]["fps"])
    if any(float(item[2]["fps"]) != fps for item in loaded):
        raise ValueError("All clips must have the same FPS")
    correction_frames = round(correction_s * fps)
    combined = loaded[0][3].copy()
    joins = []
    for tag, source, _, frames, report in loaded[1:]:
        incoming_boundary = boundary_frame(report)
        i, j, match = choose_overlap(combined, frames, incoming_boundary, fps)
        previous = combined[:i + 1]
        aligned = align_incoming(frames[j:], previous[-1])
        before = previous[-1]
        raw_gap = pose_distance(before, aligned[0])
        joined = np.concatenate((previous, aligned[1:]), axis=0)
        boundary = len(previous) - 1
        combined = inertialize_join(joined, boundary, correction_frames, fps)
        joins.append({
            "into": tag,
            "chain_boundary_frame": boundary,
            "chain_boundary_time_s": boundary / fps,
            "incoming_source_frame": j,
            "incoming_source_time_s": j / fps,
            "incoming_skill_boundary_time_s": incoming_boundary / fps,
            "raw_joint_pose_gap_rad": raw_gap,
            "corrected_first_joint_step_rad": pose_distance(combined[boundary], combined[boundary + 1]),
            "corrected_first_root_step_m": float(np.linalg.norm(combined[boundary + 1, :3] - combined[boundary, :3])),
            **match,
        })
    if not np.isfinite(combined).all():
        raise ValueError("Non-finite output")
    duration_s = (len(combined) - 1) / fps
    cfg = yaml.safe_load((ROOT / "configs/envs/transition_inertial_walk_stop_hold5s_probe.yaml").read_text())
    cfg["motion_file"] = str(output.resolve())
    cfg["episode_length"] = float(np.ceil(duration_s + 2.0))
    cfg["rand_reset"] = False
    report = {
        "purpose": "one-episode kinematic transition-chain probe for the fixed PPO tracker",
        "sequence": list(tags),
        "sources": [{"tag": tag, "file": str(source.resolve()),
                     "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
                    for tag, source, *_ in loaded],
        "method": "match overlapping source phases by joint pose/velocity; align XY and heading; decay seam pose/velocity offset over 0.3s",
        "correction_duration_s": correction_frames / fps,
        "fps": fps,
        "frames": len(combined),
        "duration_s": duration_s,
        "hold_start_s": duration_s - 5.0,
        "joins": joins,
        "status": "kinematic diagnostic only; physical tracking and naturalness unverified",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    env_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"loop_mode": 0, "fps": fps,
                     "frames": combined.astype(np.float32).tolist()}, stream)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    env_output.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn", choices=("left", "right"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-output", type=Path, required=True)
    parser.add_argument("--correction-s", type=float, default=0.3)
    args = parser.parse_args()
    print(json.dumps(build(args.turn, args.output, args.env_output,
                           args.correction_s), indent=2))
