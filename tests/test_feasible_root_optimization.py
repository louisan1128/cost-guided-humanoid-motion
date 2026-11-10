import sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from humanoid_imitation.planning.feasible_root_optimization import integrated_root,optimize_feasible_root,feasibility_cost,LIMITS


def fixture():
    r=torch.zeros(2,60,71,dtype=torch.float64);r[:,:,2]=.9;r[:,:,3]=1;r[:,:,7]=1;r[:,:,9]=1;r[:,:,0]=torch.arange(1,61)/30
    return r,torch.tensor([[1.,0.],[.5,0.]],dtype=torch.float64),torch.zeros(2,dtype=torch.float64)


def test_large_controls_obey_bounds_and_coordinate_differences():
    r,v,w=fixture();x=integrated_root(r,torch.randn(2,9,3,dtype=r.dtype)*100,v,w)
    dv=torch.diff(torch.cat((v[:,None],x[:,:,9:11]),1),dim=1)*30
    dw=torch.diff(torch.cat((w[:,None],x[:,:,14]),1),dim=1)*30
    assert x[:,:,9:11].norm(dim=-1).max()<=LIMITS['speed']+1e-8
    assert dv.norm(dim=-1).max()<=LIMITS['acceleration']+1e-8
    assert x[:,:,14].abs().max()<=LIMITS['yaw_rate']+1e-8
    assert dw.abs().max()<=LIMITS['yaw_acceleration']+1e-8
    assert torch.allclose(torch.diff(torch.cat((torch.zeros_like(x[:,:1,:2]),x[:,:,:2]),1),dim=1)*30,x[:,:,9:11])
    assert torch.allclose(x[:,:,3:6].norm(dim=-1),torch.ones_like(x[:,:,0]))


def test_directional_gradient_and_nonmutating_cost_descent():
    r,v,w=fixture();control=(torch.randn(2,9,3,dtype=r.dtype)*.1).requires_grad_()
    f=lambda c: integrated_root(r,c,v,w)[:,:,1].square().sum()+integrated_root(r,c,v,w)[:,:,0].sum()
    grad=torch.autograd.grad(f(control),control)[0];direction=torch.randn_like(control);eps=1e-5
    fd=(f(control+eps*direction)-f(control-eps*direction))/(2*eps)
    assert torch.allclose((grad*direction).sum(),fd,rtol=1e-5,atol=1e-6)
    before=r.clone();cost=lambda z:(z[:,:,1]-.25).square().mean(1)+feasibility_cost(z,v,w)
    initial=integrated_root(r,torch.zeros_like(control),v,w)
    out=optimize_feasible_root(r,cost,torch.tensor([True,False]),v,w,steps=8)
    assert cost(out)[0]<cost(initial)[0]
    assert torch.equal(out[1],r[1]) and torch.equal(r,before)
    assert torch.isfinite(out).all()
