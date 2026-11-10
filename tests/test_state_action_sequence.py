import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from humanoid_imitation.planning.state_action_sequence import *

def episode(n=12):
    s=dict(root_pos=torch.randn(n,3),root_rot=torch.tensor([[0.,0.,0.,1.]]).repeat(n,1),
           root_vel=torch.randn(n,3),root_ang_vel=torch.randn(n,3),q=torch.randn(n,28),dq=torch.randn(n,28))
    return dict(obs=torch.randn(n,144),applied_action=torch.randn(n,28),teacher_action=torch.ones(n,28)*999,pre=s,next=s)

def test_local_frame_and_quaternion_sign():
    e=episode();s=e['pre'];s['root_pos']=torch.tensor([[3.,4.,1.]]).repeat(12,1)
    a=torch.full((12,),torch.pi/2);o=torch.tensor([3.,4.,0.])
    z=local_state(s,o,a)
    assert torch.allclose(z[:,:3],torch.tensor([0.,0.,1.]).repeat(12,1),atol=1e-6)
    assert torch.allclose(rotate_z(rotate_z(s['root_vel'],a),-a),s['root_vel'],atol=1e-6)
    q=s['root_rot'];s['root_rot']=-q
    assert torch.allclose(z,local_state(s,o,a),atol=1e-6)
    assert z.shape==(12,71)

def test_windows_causal_and_applied_not_label():
    e=episode();x,y,ids=build_windows(e,8,joint=True)
    assert len(ids)==5 and y.shape==(5,8,99)
    assert torch.equal(y[:,:,:28],torch.stack([e['applied_action'][i:i+8]for i in range(5)]))
    x0=x[0].clone();e['obs'][1:]+=1000;e['next']['q'][1:]+=1000
    assert torch.equal(build_windows(e,8,joint=True)[0][0],x0)
    assert not len(valid_starts(3,8))

def test_guidance_zero_identity_and_finite():
    torch.manual_seed(2);m=SequenceNet(10,12,'diffusion',32)
    c=torch.randn(3,10);noise=torch.randn(3,12);a=alpha_schedule()
    z=sample_sequence(m,c,a,noise,4)
    assert torch.equal(z,sample_sequence(m,c,a,noise,4,lambda z:z.square().sum(-1),0))
    g=sample_sequence(m,c,a,noise,4,lambda z:z.square().sum(-1),.1)
    assert torch.isfinite(g).all() and not torch.equal(z,g)
    u=sample_sequence(m,c,a,noise,4,lambda z:z.square().sum(-1),.1,unit_gradient=True)
    assert torch.isfinite(u).all() and not torch.equal(z,u)

def test_terminal_reset_crossing_is_rejected():
    import pytest
    e=episode();e['done']=torch.zeros(12,dtype=torch.long);e['done'][5]=1
    with pytest.raises(ValueError,match='terminal'):build_windows(e,8,joint=True)

def test_execution_buffer_uses_consecutive_actions_and_replans():
    import pytest
    b=PlanExecutionBuffer(2);seq=torch.arange(4.).reshape(1,4,1).expand(1,4,99)
    assert b.needs_plan
    b.set(seq)
    assert b.take()[0,0]==0 and not b.needs_plan
    assert b.take()[0,0]==1 and b.needs_plan
    with pytest.raises(ValueError):b.take()
    b.set(seq+10);assert b.take()[0,0]==10

def test_heading_cost_wrap_and_forward_progress():
    import math
    def endpoint(angle,x=1.):
        v=torch.zeros(1,99);v[:,28]=x
        v[:,31]=math.cos(angle);v[:,32]=math.sin(angle)
        return v
    base=endpoint(math.radians(175));correct=endpoint(math.radians(-170))[:,None]
    assert heading_guidance_cost(correct,base,math.radians(15)).abs().max()<1e-6
    wrong=endpoint(math.radians(160))[:,None].requires_grad_(True)
    cost=heading_guidance_cost(wrong,base,math.radians(15))
    assert cost.item()>0
    assert torch.isfinite(torch.autograd.grad(cost.sum(),wrong)[0]).all()

def test_cached_execution_still_records_every_actual_observation():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from run_state_action_sequence import SequencePolicy
    p=SequencePolicy.__new__(SequencePolicy)
    p.hist=[];p.actions=[];p.execution=PlanExecutionBuffer(4)
    p.execution.set(torch.zeros(1,30,99))
    for i in range(4):p(torch.full((1,144),float(i)))
    assert p.execution.needs_plan
    assert [float(v[0,0])for v in p.hist]==[0,1,2,3]
    assert len(p.actions)==3 and p.plan_age==3
