#!/usr/bin/env python3
"""View one no-reset transition chain with the 500-iteration PPO tracker."""

import argparse
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "outputs/ppo_transition_stop_hold5s_500_seed0/model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("turn", choices=("left", "right"))
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--mimickit-root", type=Path, default=Path("/home/an/Desktop/MimicKit-main"))
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    config = ROOT / "configs/envs" / f"chain_start_walk_{args.turn}_walk_stop_hold5s_v1.yaml"
    model = args.model.resolve()
    mimickit = args.mimickit_root.resolve()
    for path in (config, model, mimickit / "mimickit/run.py"):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    cmd = [sys.executable, str(mimickit / "mimickit/run.py"),
           "--arg_file", "args/deepmimic_humanoid_ppo_args.txt",
           "--mode", "test", "--num_envs", "1", "--visualize", "true",
           "--test_episodes", str(args.episodes), "--rand_seed", str(args.seed),
           "--env_config", str(config), "--model_file", str(model)]
    if args.dry_run:
        print("cwd:", mimickit)
        print(shlex.join(cmd))
        return
    with tempfile.TemporaryDirectory(prefix="continuous_chain_view_") as out_dir:
        subprocess.run(cmd + ["--out_dir", out_dir], cwd=mimickit, check=True)


if __name__ == "__main__":
    main()
