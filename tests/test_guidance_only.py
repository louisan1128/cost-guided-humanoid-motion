import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from humanoid_imitation.planning.guidance_only import guided_sample, task_cost, risk_gate
from humanoid_imitation.planning.state_action_sequence import sample_sequence

class Model:
    method='diffusion'
    def __call__(self,context,x,t):
        return .4*x+.01*context[:,:x.shape[1]]


def test_disabled_and_legacy_sampler_match_existing_output_exactly():
    torch.manual_seed(1);x=torch.randn(2,12);c=torch.randn(2,12);a=torch.linspace(.99,.01,100);m=Model();cost=lambda z:z.square().mean(1)
    for scale in [0.,3.]:
        assert torch.equal(guided_sample(m,c,a,x,cost,scale,'legacy'),sample_sequence(m,c,a,x,16,cost,scale,True))
    assert torch.equal(guided_sample(m,c,a,x,cost,0.,'clip'),sample_sequence(m,c,a,x))


def scene():
    r=torch.zeros(1,60,71,dtype=torch.float64);r[:,:,0]=torch.arange(1,61)/30;r[:,:,3]=1;r[:,:,7]=1;r[:,:,9]=1
    return r,torch.tensor([[1.,.3]],dtype=torch.float64),torch.tensor([[.25,.4]],dtype=torch.float64),torch.zeros(1,dtype=torch.float64),torch.tensor([False]),torch.tensor([[6.,0.]],dtype=torch.float64)


def test_task_cost_gradient_matches_finite_difference_and_geometry_mirror():
    r,c,h,a,d,g=scene();nom=r.clone();r.requires_grad_();f=lambda x:task_cost(x,nom,c,h,a,d,g).sum();gradient=torch.autograd.grad(f(r),r)[0];direction=torch.zeros_like(r);direction[:,:,1]=1
    epsilon=1e-5;fd=(f(r.detach()+epsilon*direction)-f(r.detach()-epsilon*direction))/(2*epsilon)
    assert torch.allclose(fd,(gradient*direction).sum(),rtol=1e-4,atol=1e-4)
    rr=r.detach().clone();rr[:,:,1]*=-1;rr[:,:,10]*=-1;cc=c.clone();cc[:,1]*=-1
    assert torch.allclose(task_cost(r,nom,c,h,a,d,g),task_cost(rr,nom,cc,h,a,d,g))


def test_geometry_gate_disables_only_safe_plans():
    r,c,h,a,d,g=scene();assert risk_gate(r,c,h,a,d).item()
    assert not risk_gate(r,torch.tensor([[20.,20.]]),h,a,d).item()


def test_zero_cost_preserves_sampler_and_no_input_mutation():
    torch.manual_seed(2);x=torch.randn(2,12);original=x.clone();c=torch.randn(2,12);a=torch.linspace(.99,.01,100);m=Model();trace=[]
    p=guided_sample(m,c,a,x,lambda z:(z*0).sum(1),30.,'clip',diagnostics=trace)
    assert torch.equal(p,sample_sequence(m,c,a,x));assert torch.equal(x,original)
    assert len(trace)==16 and all(v['max_update_norm']==0 for v in trace)
