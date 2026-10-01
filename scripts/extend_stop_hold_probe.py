#!/usr/bin/env python3
"""Extend an already stationary stop reference for a physical tracking probe."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np
import yaml

from build_start_walk_probe import load_motion


def build(source: Path, output: Path, env_template: Path, env_output: Path,
          existing_hold_s: float, extra_hold_s: float) -> dict:
    for path in (output, output.with_suffix(".json"), env_output):
        if path.exists():
            raise FileExistsError(path)
    if existing_hold_s <= 0 or extra_hold_s <= 0:
        raise ValueError("Hold durations must be positive")
    motion, frames = load_motion(source)
    fps = float(motion["fps"])
    existing_frames = round(existing_hold_s * fps)
    extra_frames = round(extra_hold_s * fps)
    if existing_frames >= len(frames) or extra_frames < 1:
        raise ValueError("Invalid hold frame count")
    if not np.allclose(frames[-existing_frames - 1:], frames[-1], atol=1e-6, rtol=0):
        raise ValueError("Source end is not a static pose for the claimed hold duration")

    extended = np.concatenate((frames, np.repeat(frames[-1:], extra_frames, axis=0)))
    source_report = json.loads(source.with_suffix(".json").read_text())
    report = dict(source_report)
    report.update({
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "method": "append identical final pose frames for a stop-hold diagnostic",
        "existing_hold_s": existing_frames / fps,
        "extra_hold_s": extra_frames / fps,
        "hold_start_s": (len(frames) - existing_frames - 1) / fps,
        "duration_s": (len(extended) - 1) / fps,
        "status": "kinematic diagnostic reference; physical hold unverified",
    })
    cfg = yaml.safe_load(env_template.read_text())
    if Path(cfg["motion_file"]).resolve() != source.resolve():
        raise ValueError("Environment template points to a different reference")
    cfg["motion_file"] = str(output.resolve())
    cfg["episode_length"] = max(float(cfg["episode_length"]), report["duration_s"] + 1.0)

    output.parent.mkdir(parents=True, exist_ok=True)
    env_output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"loop_mode": 0, "fps": fps,
                     "frames": extended.astype(np.float32).tolist()}, stream)
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    env_output.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-template", type=Path, required=True)
    parser.add_argument("--env-output", type=Path, required=True)
    parser.add_argument("--existing-hold-s", type=float, default=2.0)
    parser.add_argument("--extra-hold-s", type=float, default=3.0)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output, args.env_template,
                           args.env_output, args.existing_hold_s,
                           args.extra_hold_s), indent=2))
