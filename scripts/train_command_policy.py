"""Launch the command-conditioned PPO + AMP policy through an external MimicKit checkout."""

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
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
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
        default=PROJECT_ROOT / "configs" / "command" / "env.yaml",
    )
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs" / "command_amp_seed0")
    parser.add_argument(
        "--agent-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "command" / "agent.yaml",
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

    import yaml
    agent_settings = yaml.safe_load(agent_config.read_text())
    env_settings = yaml.safe_load(env_config.read_text())
    if env_settings['env_name'] != 'command_amp' or agent_settings['agent_name'] != 'AMP':
        raise ValueError("This launcher requires command_amp and AMP configs")
    if args.mode == 'test' and model_file is None:
        raise ValueError("--model-file is required in test mode")
    # Source configs use project-relative paths. MimicKit runs from its own
    # checkout, so runtime configs below contain resolved absolute paths.
    manifest = Path(env_settings['motion_file'])
    if not manifest.is_absolute():
        manifest = PROJECT_ROOT / manifest
    manifest_settings = yaml.safe_load(manifest.read_text())
    for row in manifest_settings['motions']:
        motion = Path(row['file'])
        if not motion.is_absolute():
            motion = PROJECT_ROOT / motion
        if not motion.is_file():
            raise FileNotFoundError(motion)
        row['file'] = str(motion.resolve())
    if args.iterations <= 0 or args.num_envs <= 0:
        raise ValueError("iterations and num-envs must be positive")
    rollout_samples = args.num_envs * agent_settings["steps_per_iter"]
    if args.mode == "train" and agent_settings["disc_buffer_size"] < rollout_samples:
        raise ValueError(
            f"AMP disc_buffer_size={agent_settings['disc_buffer_size']} is too small: "
            f"num_envs * steps_per_iter = {rollout_samples}. "
            "Increase disc_buffer_size in the agent config or reduce --num-envs."
        )
    args.max_samples = args.iterations * rollout_samples
    if args.mode == "train" and args.out_dir.exists() and any(args.out_dir.iterdir()):
        raise ValueError("Output directory is not empty; choose a new --out-dir")
    if args.dry_run:
        print(f"{args.mode}: {args.num_envs} envs, {args.iterations} iterations, {args.max_samples} samples")
        print(f"env={env_config}\nagent={agent_config}\noutput={args.out_dir.resolve()}")
        return
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
    sys.path.insert(0, str(mimickit_package_root))

    import envs.env_builder as env_builder
    from humanoid_imitation.envs.command_amp_env import (
        CommandAMPEnv,
    )
    import run as mimickit_run

    native_build_env = env_builder.build_env

    def build_project_env(env_file, engine_file, num_envs, device, visualize, record_video=False):
        env_config, engine_config = env_builder.load_configs(env_file, engine_file)
        if env_config["env_name"] != "command_amp":
            return native_build_env(
                env_file, engine_file, num_envs, device, visualize, record_video
            )
        return CommandAMPEnv(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )

    env_builder.build_env = build_project_env
    args.out_dir.mkdir(parents=True, exist_ok=True)
    runtime_manifest = args.out_dir.resolve() / 'runtime_motion_manifest.yaml'
    runtime_env = args.out_dir.resolve() / 'runtime_env_config.yaml'
    runtime_manifest.write_text(yaml.safe_dump(manifest_settings, sort_keys=False))
    env_settings['motion_file'] = str(runtime_manifest)
    runtime_env.write_text(yaml.safe_dump(env_settings, sort_keys=False))
    if args.mode == 'train':
        import json
        (args.out_dir / 'motion_manifest.yaml').write_text(manifest.read_text())
        (args.out_dir / 'launch.json').write_text(json.dumps({
            'iterations': args.iterations, 'num_envs': args.num_envs,
            'max_samples': args.max_samples, 'seed': args.seed,
            'model_file': str(model_file) if model_file else None,
            'initialization': 'weights_only' if model_file else 'fresh',
        }, indent=2))

    run_argv = [
        str(mimickit_package_root / "run.py"),
        "--mode", args.mode,
        "--num_envs", str(args.num_envs),
        "--devices", args.device,
        "--visualize", str(args.visualize).lower(),
        "--max_samples", str(args.max_samples),
        "--rand_seed", str(args.seed),
        "--engine_config", str(engine_config),
        "--env_config", str(runtime_env),
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
