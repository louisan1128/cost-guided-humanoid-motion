#!/usr/bin/env python3
"""Replace the all-NaN neck expmap in MimicKit's generic humanoid walk."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle

import numpy as np


def repair(source: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing motion: {output}")
    with source.open("rb") as stream:
        motion = pickle.load(stream)
    frames = np.asarray(motion["frames"], dtype=np.float32)
    if frames.ndim != 2 or frames.shape[1] != 34:
        raise ValueError(f"Expected generic humanoid 34-D frames, got {frames.shape}")
    bad = ~np.isfinite(frames)
    expected = np.zeros_like(bad)
    expected[:, 9:12] = True  # Neck 3-D exponential-map rotation.
    if not np.array_equal(bad, expected):
        raise ValueError("Expected only the three neck values to be NaN in every frame")
    frames[:, 9:12] = 0.0  # Neutral neck; all other source values are unchanged.
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump({"loop_mode": motion["loop_mode"], "fps": motion["fps"],
                     "frames": frames.tolist()}, stream)
    report = {"source": str(source.resolve()),
              "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "modified": "all NaN neck expmap values at frame columns 9:12 set to 0",
              "changed_values": int(bad.sum()), "frames": len(frames),
              "fps": float(motion["fps"]), "loop_mode": int(motion["loop_mode"]),
              "status": "reference repair; PPO tracking not yet tested"}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(json.dumps(repair(args.source, args.output), indent=2))
