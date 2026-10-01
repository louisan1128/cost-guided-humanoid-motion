#!/usr/bin/env python3
"""Copy an exact, frame-aligned time interval from a MimicKit motion PKL."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle


def crop(source: Path, output: Path, start_s: float, end_s: float) -> dict:
    if start_s < 0 or end_s <= start_s:
        raise ValueError("Expected 0 <= start_s < end_s")
    if output.exists():
        raise FileExistsError(f"Will not overwrite existing motion: {output}")
    with source.open("rb") as stream:
        motion = pickle.load(stream)
    fps = float(motion["fps"])
    frames = motion["frames"]
    first, last = round(start_s * fps), round(end_s * fps)
    if abs(first / fps - start_s) > 1e-5 or abs(last / fps - end_s) > 1e-5:
        raise ValueError("Crop boundaries must lie on source frame times")
    if first < 0 or last >= len(frames):
        raise ValueError("Crop exceeds source motion duration")

    # Include both boundary frames: (last - first) / fps is the requested duration.
    cropped = {"loop_mode": 0, "fps": fps, "frames": frames[first:last + 1]}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        pickle.dump(cropped, stream)
    manifest = {
        "source": str(source.resolve()),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "start_s": start_s, "end_s": end_s,
        "first_frame": first, "last_frame": last,
        "fps": fps, "frames": len(cropped["frames"]),
        "duration_s": (len(cropped["frames"]) - 1) / fps,
        "root_coordinates": "preserved from source; no rebasing",
        "status": "kinematic crop; physical PPO tracking not tested",
    }
    output.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start", type=float, required=True)
    parser.add_argument("--end", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(crop(args.source, args.output, args.start, args.end), indent=2))
