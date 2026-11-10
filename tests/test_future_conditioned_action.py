import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch import nn
from humanoid_imitation.planning.future_conditioned_action import root_features,fixed_travel_observation,FutureConditionedAction


def test_features_ignore_future_joints_and_keep_selected_root_samples():
 x=torch.randn(3,30,71);y=x.clone();y[:,:,15:]+=100
 assert root_features(x).shape==(3,32)
 assert torch.equal(root_features(x),root_features(y))
 y[:,29,0]+=1
 assert not torch.equal(root_features(x),root_features(y))


def test_task_reexpression_preserves_physical_observation_and_rotates_goal():
 obs=torch.randn(2,144);pos=torch.zeros(2,3);yaw=torch.tensor([0.,torch.pi/2]);goal=torch.tensor([[5.,0.],[5.,0.]])
 out=fixed_travel_observation(obs,pos,yaw,goal)
 assert torch.equal(out[:,:140],obs[:,:140])
 assert torch.allclose(out[0,-4:],torch.tensor([1.,0.,0.,1.]),atol=1e-6)
 assert torch.allclose(out[1,-4:],torch.tensor([0.,-1.,-1.,0.]),atol=1e-6)


def test_decoder_starts_at_prior_and_can_learn_future_dependence_without_prior_grad():
 prior=nn.Linear(144,28);m=FutureConditionedAction(prior,torch.ones(28),torch.zeros(32),torch.ones(32));obs=torch.randn(2,144);f=torch.randn(2,32)
 assert torch.equal(m(obs,f),m.prior(obs))
 with torch.no_grad():m.residual[-1].weight.normal_(std=.01)
 f.requires_grad_(True);out=m(obs,f);out.sum().backward()
 assert f.grad.abs().sum()>0
 assert all(p.grad is None for p in m.prior.parameters())
 assert ((out-m.prior(obs)).abs()<=.5).all()
 # Task-goal changes are excluded from the learned correction.
 obs2=obs.clone();obs2[:,-4:]+=2
 assert torch.allclose(m(obs,f)-m.prior(obs),m(obs2,f)-m.prior(obs2),atol=1e-6)


def test_goal_aware_correction_can_distinguish_existing_task_from_future_intent():
 m=FutureConditionedAction(nn.Linear(144,28),torch.ones(28),torch.zeros(32),torch.ones(32),include_goal=True)
 with torch.no_grad():m.residual[-1].weight.normal_(std=.01)
 obs=torch.randn(2,144,requires_grad=True);feature=torch.randn(2,32,requires_grad=True)
 correction=(m(obs,feature)-m.prior(obs)).sum();go,gr=torch.autograd.grad(correction,(obs,feature))
 assert go[:,-4:].abs().sum()>0
 assert gr.abs().sum()>0
