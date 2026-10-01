"""Open MimicKit's G1 reference walking motion in the Isaac Lab GUI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimickit-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    mimickit_root = args.mimickit_root.resolve()
    package_root = mimickit_root / "mimickit"
    if not (package_root / "run.py").is_file():
        raise FileNotFoundError(f"Expected MimicKit run.py at {package_root / 'run.py'}")

    sys.path.insert(0, str(package_root))
    import run as mimickit_run

    argv = [
        str(package_root / "run.py"),
        "--mode", "test",
        "--num_envs", "1",
        "--devices", args.device,
        "--visualize", "true",
        "--test_episodes", "1000000",
        "--engine_config", str(PROJECT_ROOT / "configs" / "engines" / "isaac_lab.yaml"),
        "--env_config", str(mimickit_root / "data" / "envs" / "view_motion_g1_env.yaml"),
    ]

    previous_cwd = Path.cwd()
    try:
        os.chdir(mimickit_root)
        mimickit_run.main(argv)
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
