"""Speed-conditioned DeepMimic environment for a PPO expert.

This module is loaded by ``scripts/train_speed_expert.py`` after MimicKit's
``mimickit`` directory has been added to ``sys.path``.
"""

import torch

import envs.deepmimic_env as deepmimic_env
import util.torch_util as torch_util


class SpeedTrackingDeepMimicEnv(deepmimic_env.DeepMimicEnv):
    """Add a forward-speed command and reward to MimicKit's DeepMimic task.

    The imitation term keeps the initial policy close to the walking reference.
    The speed term asks the policy to match a commanded heading-local forward
    velocity. This is the first PPO curriculum stage, not the final reward design.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._speed_window = None
        self._target_speed_label = None
        self._measured_speed_label = None
        self._speed_error_label = None

        self._target_speed_min = float(env_config["target_speed_min"])
        self._target_speed_max = float(env_config["target_speed_max"])
        self._command_change_time_min = float(env_config["command_change_time_min"])
        self._command_change_time_max = float(env_config["command_change_time_max"])
        self._track_global_root_override = env_config.get("track_global_root")
        self._speed_reward_weight = float(env_config["speed_reward_weight"])
        self._speed_error_scale = float(env_config["speed_error_scale"])
        self._lateral_error_scale = float(env_config["lateral_error_scale"])

        assert 0.0 <= self._target_speed_min <= self._target_speed_max
        assert 0.0 < self._command_change_time_min <= self._command_change_time_max
        assert 0.0 <= self._speed_reward_weight <= 1.0
        assert self._speed_error_scale > 0.0
        assert self._lateral_error_scale >= 0.0

        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )

        if self._visualize:
            self._build_speed_monitor()

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)
        num_envs = self.get_num_envs()
        self._target_speed = torch.zeros(num_envs, device=self._device, dtype=torch.float32)
        self._command_change_times = torch.zeros(num_envs, device=self._device, dtype=torch.float32)

    def _track_global_root(self):
        if self._track_global_root_override is None:
            return super()._track_global_root()
        return bool(self._track_global_root_override)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_speed_command(env_ids)

    def _reset_speed_command(self, env_ids):
        n = len(env_ids)
        speed_sample = torch.rand(n, device=self._device)
        speed_sample = (
            self._target_speed_min
            + (self._target_speed_max - self._target_speed_min) * speed_sample
        )
        self._target_speed[env_ids] = speed_sample

        interval_sample = torch.rand(n, device=self._device)
        interval_sample = (
            self._command_change_time_min
            + (self._command_change_time_max - self._command_change_time_min) * interval_sample
        )
        self._command_change_times[env_ids] = self._time_buf[env_ids] + interval_sample

    def _update_misc(self):
        super()._update_misc()
        change_env_ids = (self._time_buf >= self._command_change_times).nonzero(
            as_tuple=False
        ).flatten()
        if len(change_env_ids) > 0:
            self._reset_speed_command(change_env_ids)

        if self._visualize:
            self._update_speed_monitor()

    def _build_speed_monitor(self):
        import omni.ui as ui

        self._speed_window = ui.Window("Speed Monitor", width=300, height=145)
        with self._speed_window.frame:
            with ui.VStack(spacing=6, style={"font_size": 18}):
                ui.Label("Environment 0", height=24)
                self._target_speed_label = ui.Label("Target:   -- m/s", height=24)
                self._measured_speed_label = ui.Label("Measured: -- m/s", height=24)
                self._speed_error_label = ui.Label("Abs error: -- m/s", height=24)

        self._update_speed_monitor()

    def _update_speed_monitor(self):
        if self._target_speed_label is None:
            return

        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        measured_speed = torch_util.quat_rotate(heading_inv_rot, root_vel)[0, 0].item()
        target_speed = self._target_speed[0].item()
        speed_error = abs(measured_speed - target_speed)

        self._target_speed_label.text = f"Target:   {target_speed:5.2f} m/s"
        self._measured_speed_label.text = f"Measured: {measured_speed:5.2f} m/s"
        self._speed_error_label.text = f"Abs error: {speed_error:5.2f} m/s"

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        target_speed = self._target_speed
        if env_ids is not None:
            target_speed = target_speed[env_ids]
        return torch.cat([obs, target_speed.unsqueeze(-1)], dim=-1)

    def _update_reward(self):
        super()._update_reward()
        imitation_reward = self._reward_buf.clone()

        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)

        speed_reward = compute_speed_tracking_reward(
            local_root_vel=local_root_vel,
            target_speed=self._target_speed,
            speed_error_scale=self._speed_error_scale,
            lateral_error_scale=self._lateral_error_scale,
        )
        imitation_weight = 1.0 - self._speed_reward_weight
        self._reward_buf[:] = (
            imitation_weight * imitation_reward
            + self._speed_reward_weight * speed_reward
        )

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)

        target_speed = self._target_speed
        measured_speed = local_root_vel[..., 0]
        if env_ids is not None:
            target_speed = target_speed[env_ids]
            measured_speed = measured_speed[env_ids]

        self._info["target_forward_speed"] = target_speed
        self._info["measured_forward_speed"] = measured_speed

    def record_diagnostics(self):
        diagnostics = super().record_diagnostics()
        char_id = self._get_char_id()
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        heading_inv_rot = torch_util.calc_heading_quat_inv(root_rot)
        local_root_vel = torch_util.quat_rotate(heading_inv_rot, root_vel)
        measured_speed = local_root_vel[..., 0]

        diagnostics["target_speed"] = torch.mean(self._target_speed)
        diagnostics["measured_speed"] = torch.mean(measured_speed)
        diagnostics["speed_mae"] = torch.mean(
            torch.abs(measured_speed - self._target_speed)
        )
        return diagnostics


@torch.jit.script
def compute_speed_tracking_reward(
    local_root_vel: torch.Tensor,
    target_speed: torch.Tensor,
    speed_error_scale: float,
    lateral_error_scale: float,
) -> torch.Tensor:
    """Reward target forward velocity while suppressing lateral drift."""

    forward_error = local_root_vel[..., 0] - target_speed
    lateral_error = local_root_vel[..., 1]
    error = speed_error_scale * torch.square(forward_error)
    error = error + lateral_error_scale * torch.square(lateral_error)
    return torch.exp(-error)
