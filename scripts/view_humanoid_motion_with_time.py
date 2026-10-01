#!/usr/bin/env python3
"""Show a MimicKit humanoid reference with a live time panel in Isaac Sim.

This is kinematic reference playback, not PPO or physical tracking.
"""

import argparse
import os
import pickle
from pathlib import Path
import sys
import tempfile

import numpy as np
import yaml


DEFAULT_MIMICKIT = Path("/home/an/Desktop/MimicKit-main")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mimickit-root", type=Path, default=DEFAULT_MIMICKIT)
    parser.add_argument("--motion-file", type=Path, required=True)
    parser.add_argument("--focus-start", type=float, default=3.0)
    parser.add_argument("--focus-end", type=float, default=5.0)
    parser.add_argument("--source-time-offset", type=float, default=0.0,
                        help="Add this offset when displaying time for a cropped clip")
    parser.add_argument("--static-camera", action="store_true",
                        help="Keep a side camera centered on the motion path")
    args = parser.parse_args()
    if args.focus_start < 0 or args.focus_end <= args.focus_start:
        parser.error("Expected 0 <= --focus-start < --focus-end")

    mimickit = args.mimickit_root.resolve()
    motion_file = args.motion_file.resolve()
    if not motion_file.is_file():
        parser.error(f"Motion file does not exist: {motion_file}")
    config_path = mimickit / "data/envs/view_motion_humanoid_env.yaml"
    if not config_path.is_file():
        parser.error(f"MimicKit config does not exist: {config_path}")

    # Preserve the external MimicKit files. Only this launch uses a temporary config.
    config = yaml.safe_load(config_path.read_text())
    config["motion_file"] = str(motion_file)
    if args.static_camera:
        config["camera_mode"] = "still"
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="view_motion_time_",
                                     dir="/tmp", delete=False) as temp:
        yaml.safe_dump(config, temp)
        temp_config = Path(temp.name)

    sys.path.insert(0, str(mimickit / "mimickit"))
    import envs.view_motion_env as view_motion_env
    import run as mimickit_run

    original_render = view_motion_env.ViewMotionEnv._render_scene
    if args.static_camera:
        with motion_file.open("rb") as stream:
            motion = pickle.load(stream)
        frames = np.asarray(motion["frames"])
        center_xy = 0.5 * (frames[0, :2] + frames[-1, :2])
        original_build_camera = view_motion_env.ViewMotionEnv._build_camera

        def build_centered_camera(env, env_config):
            original_build_camera(env, env_config)
            target = np.array([center_xy[0], center_xy[1], 1.0])
            env._camera.lookat(target + np.array([0.0, -6.0, 2.0]), target)

        view_motion_env.ViewMotionEnv._build_camera = build_centered_camera

    def render_with_time(env):
        # The Isaac Sim app exists by the first render, so import its UI here.
        if not hasattr(env, "_motion_time_window"):
            import omni.ui as ui

            env._motion_time_window = ui.Window("Motion time", width=300, height=105)
            with env._motion_time_window.frame:
                with ui.VStack(spacing=6):
                    env._motion_time_label = ui.Label("Motion time: 0.00 s", height=30)
                    env._motion_focus_label = ui.Label("", height=30)

        t = float(env.get_env_time()[0].item()) * float(env._time_scale)
        shown_time = t + args.source_time_offset
        env._motion_time_label.text = f"Source time: {shown_time:.2f} s"
        marker = "<-- inspect now" if args.focus_start <= shown_time <= args.focus_end else ""
        env._motion_focus_label.text = (
            f"Turn: {args.focus_start:g}-{args.focus_end:g} s {marker}"
        )
        original_render(env)

    view_motion_env.ViewMotionEnv._render_scene = render_with_time
    argv = [str(mimickit / "mimickit/run.py"),
            "--arg_file", "args/view_motion_humanoid_args.txt",
            "--env_config", str(temp_config),
            "--mode", "test", "--num_envs", "1", "--visualize", "true",
            "--test_episodes", "1000000", "--out_dir", "/tmp/view_motion_with_time"]
    previous_cwd = Path.cwd()
    try:
        os.chdir(mimickit)
        mimickit_run.main(argv)
    finally:
        os.chdir(previous_cwd)
        temp_config.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
