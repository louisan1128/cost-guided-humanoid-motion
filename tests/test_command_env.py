"""CPU tests: command timing without launching the simulator."""
import os
import sys
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
r = Path(__file__).resolve().parents[1]
mimickit_root = Path(os.environ.get('MIMICKIT_ROOT', str(r.parent / 'MimicKit-main')))
sys.path[:0] = [str(r / 'src'), str(mimickit_root / 'mimickit')]
from humanoid_imitation.envs.command_amp_env import CommandAMPEnv, AMPEnv, task_reward

class CommandTests(unittest.TestCase):
    def make_env(self):
        e = object.__new__(CommandAMPEnv)
        e._device = 'cpu'
        e.command_config = dict(command_vx_range=[.7,1.3], command_wz_range=[-.5,.5], command_interval=[2,4], stop_probability=0)
        e.command = torch.zeros(3,2)
        e.reward_command = torch.zeros_like(e.command)
        e.next_command_time = torch.zeros(3)
        e._time_buf = torch.tensor([1.,2.,3.])
        return e
    def test_reward_and_turn_sign(self):
        v = torch.tensor([[1.,0.,0.]])
        c = torch.tensor([[1.,.5]])
        self.assertAlmostEqual(task_reward(v,torch.tensor([.5]),c).item(),1)
        self.assertLess(task_reward(v,torch.tensor([-.5]),c).item(),1)
    def test_subset_and_interval(self):
        e=self.make_env(); e._sample_commands(torch.tensor([1]))
        self.assertTrue(torch.equal(e.command[[0,2]],torch.zeros(2,2)))
        self.assertTrue(4 <= e.next_command_time[1] <= 6)
    def test_boundary_rewards_old_command(self):
        e=self.make_env(); e.command[:]=torch.tensor([1.,.5])
        with patch.object(AMPEnv,'_pre_physics_step'), patch.object(AMPEnv,'_update_misc'):
            e._pre_physics_step(None)
            e._update_misc()
        self.assertTrue(torch.equal(e.reward_command, torch.tensor([[1.,.5]]).repeat(3,1)))
        self.assertTrue(torch.equal(e._time_buf,torch.tensor([1.,2.,3.])))
    def test_schedule_and_stop(self):
        e=self.make_env(); e.command_config['command_schedule']=[[0,1,0],[2,0,0],[3,1,-.4]]
        e._sample_commands(torch.arange(3))
        torch.testing.assert_close(e.command,torch.tensor([[1.,0.],[0.,0.],[1.,-.4]]))

if __name__=='__main__': unittest.main()
