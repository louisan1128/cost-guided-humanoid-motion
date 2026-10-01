"""Launch the speed-conditioned PPO expert through an external MimicKit checkout."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mimickit-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-samples", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument(
        "--engine-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "engines" / "isaac_lab.yaml",
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "envs" / "g1_speed_fixed_1p0.yaml",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "ppo_speed_stage1")
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "agents" / "ppo_g1_speed_stage1.yaml",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-file", type=Path)
    parser.add_argument("--test-episodes", type=int, default=2)
    parser.add_argument("--logger", choices=("txt", "tb", "wandb"), default="txt")
    parser.add_argument("--save-int-models", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mimickit_root = args.mimickit_root.resolve()
    mimickit_package_root = mimickit_root / "mimickit"
    if not (mimickit_package_root / "run.py").is_file():
        raise FileNotFoundError(
            f"Expected MimicKit run.py at {mimickit_package_root / 'run.py'}"
        )

    engine_config = args.engine_config.resolve()
    env_config = args.env_config.resolve()
    agent_config = args.agent_config.resolve()
    for config_path in (engine_config, env_config, agent_config):
        if not config_path.is_file():
            raise FileNotFoundError(f"Config file not found: {config_path}")

    if args.model_file is not None:
        model_file = args.model_file.resolve()
        if not model_file.is_file():
            raise FileNotFoundError(f"Model file not found: {model_file}")
    else:
        model_file = None

    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(mimickit_package_root))

    import envs.env_builder as env_builder
    from humanoid_imitation.envs.speed_tracking_deepmimic_env import (
        SpeedTrackingDeepMimicEnv,
    )
    import run as mimickit_run

    native_build_env = env_builder.build_env

    def build_project_env(env_file, engine_file, num_envs, device, visualize, record_video=False):
        env_config, engine_config = env_builder.load_configs(env_file, engine_file)
        if env_config["env_name"] != "speed_tracking_deepmimic":
            return native_build_env(
                env_file, engine_file, num_envs, device, visualize, record_video
            )
        return SpeedTrackingDeepMimicEnv(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )

    env_builder.build_env = build_project_env
    args.out_dir.mkdir(parents=True, exist_ok=True)

    run_argv = [
        str(mimickit_package_root / "run.py"),
        "--mode", args.mode,
        "--num_envs", str(args.num_envs),
        "--devices", args.device,
        "--visualize", str(args.visualize).lower(),
        "--max_samples", str(args.max_samples),
        "--rand_seed", str(args.seed),
        "--engine_config", str(engine_config),
        "--env_config", str(env_config),
        "--agent_config", str(agent_config),
        "--out_dir", str(args.out_dir.resolve()),
        "--logger", args.logger,
        "--save_int_models", str(args.save_int_models).lower(),
    ]
    if model_file is not None:
        run_argv.extend(["--model_file", str(model_file)])

    if args.mode == "test":
        if model_file is None:
            raise ValueError("--model-file is required in test mode")
        run_argv.extend(
            [
                "--test_episodes", str(args.test_episodes),
            ]
        )

    previous_cwd = Path.cwd()
    try:
        os.chdir(mimickit_root)
        mimickit_run.main(run_argv)
    finally:
        os.chdir(previous_cwd)


if __name__ == "__main__":
    main()
