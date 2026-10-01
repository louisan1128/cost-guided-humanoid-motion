#!/usr/bin/env python3
"""Rigidly align a MimicKit motion's initial root heading with world +X."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
from scipy.spatial.transform import Rotation


def align(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing motion: {output}")
    with source.open("rb") as stream:
        motion = pickle.load(stream)
    frames = np.asarray(motion["frames"], dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != 34:
        raise ValueError(f"Expected generic humanoid 34-D frames, got {frames.shape}")

    root_rot = Rotation.from_rotvec(frames[:, 3:6])
    forward = root_rot[0].apply([1.0, 0.0, 0.0])
    initial_yaw = float(np.arctan2(forward[1], forward[0]))
    world_rotation = Rotation.from_euler("z", -initial_yaw)
    first_xy = frames[0, :2].copy()
    offset_xy = frames[:, :2] - first_xy
    frames[:, :2] = world_rotation.apply(
        np.column_stack((offset_xy, np.zeros(len(frames))))
    )[:, :2]
    frames[:, 3:6] = (world_rotation * root_rot).as_rotvec()

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"loop_mode": 0, "fps": motion["fps"],
                     "frames": frames.astype(np.float32).tolist()}, stream)
    headings = (world_rotation * root_rot).apply([1.0, 0.0, 0.0])
    yaw = np.unwrap(np.arctan2(headings[:, 1], headings[:, 0]))
    report = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "operation": "rigid world-Z rotation to initial +X heading; root XY rebased to zero",
        "initial_heading_before_deg": float(np.degrees(initial_yaw)),
        "heading_change_deg": float(np.degrees(yaw[-1] - yaw[0])),
        "root_displacement_xy_m": frames[-1, :2].tolist(),
        "fps": float(motion["fps"]), "frames": len(frames),
        "status": "kinematic transform; physical PPO tracking not tested",
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(align(args.source, args.output), indent=2))
