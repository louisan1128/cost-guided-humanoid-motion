"""Task-command PPO with MimicKit AMP style reward; independent of the tracker."""
import torch
from envs.amp_env import AMPEnv
from util import torch_util


def task_reward(local_velocity, angular_z, command, linear_weight=0.7):
    linear = torch.exp(-4 * (local_velocity[:, 0] - command[:, 0]).square()
                       -4 * local_velocity[:, 1].square())
    angular = torch.exp(-4 * (angular_z - command[:, 1]).square())
    return linear_weight * linear + (1.0 - linear_weight) * angular


class CommandAMPEnv(AMPEnv):
    def __init__(self, env_config, **kwargs):
        self.command_config = env_config
        for name in ('command_vx_range', 'command_wz_range', 'command_interval'):
            lo, hi = env_config[name]
            if lo > hi or (name == 'command_interval' and lo <= 0):
                raise ValueError(name)
        if not 0 <= env_config['stop_probability'] <= 1:
            raise ValueError('stop_probability')
        if not 0 <= env_config.get('linear_reward_weight', 0.7) <= 1:
            raise ValueError('linear_reward_weight')
        super().__init__(env_config=env_config, **kwargs)

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        self.command = torch.zeros((self.get_num_envs(), 2), device=self._device)
        self.reward_command = torch.zeros_like(self.command)
        self.next_command_time = torch.zeros(self.get_num_envs(), device=self._device)

    def _sample_commands(self, ids):
        c = self.command_config
        if 'command_schedule' in c:
            schedule = c['command_schedule']
            for i, row in enumerate(schedule):
                end = schedule[i+1][0] if i+1 < len(schedule) else float('inf')
                selected = ids[(self._time_buf[ids] >= row[0]) & (self._time_buf[ids] < end)]
                self.command[selected] = torch.tensor(row[1:], device=self._device, dtype=self.command.dtype)
                self.next_command_time[selected] = end
            return
        for i, key in enumerate(('command_vx_range', 'command_wz_range')):
            lo, hi = c[key]
            self.command[ids, i] = lo + (hi-lo)*torch.rand(len(ids), device=self._device)
        stop = torch.rand(len(ids), device=self._device) < c['stop_probability']
        self.command[ids[stop]] = 0
        lo, hi = c['command_interval']
        self.next_command_time[ids] = self._time_buf[ids] + lo + (hi-lo)*torch.rand(len(ids), device=self._device)

    def _reset_envs(self, ids):
        super()._reset_envs(ids)
        if len(ids):
            self._sample_commands(ids)
            self.reward_command[ids] = self.command[ids]

    def _pre_physics_step(self, actions):
        # Reward the command the actor saw, even on the resampling boundary.
        self.reward_command.copy_(self.command)
        super()._pre_physics_step(actions)

    def _update_misc(self):
        super()._update_misc()
        ids = (self._time_buf >= self.next_command_time).nonzero().flatten()
        if len(ids):
            self._sample_commands(ids)

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        command = self.command if env_ids is None else self.command[env_ids]
        return torch.cat((obs, command), dim=-1)

    def _measured(self):
        cid = self._get_char_id()
        rotation = torch_util.calc_heading_quat_inv(self._engine.get_root_rot(cid))
        velocity = torch_util.quat_rotate(rotation, self._engine.get_root_vel(cid))
        return velocity, self._engine.get_root_ang_vel(cid)[:, 2]

    def _update_reward(self):
        velocity, angular = self._measured()
        self._reward_buf[:] = task_reward(
            velocity, angular, self.reward_command,
            linear_weight=self.command_config.get('linear_reward_weight', 0.7))

    def record_diagnostics(self):
        out = super().record_diagnostics()
        velocity, angular = self._measured()
        out['command_vx_mae'] = (velocity[:, 0]-self.reward_command[:, 0]).abs().mean()
        out['command_wz_mae'] = (angular-self.reward_command[:, 1]).abs().mean()
        return out
