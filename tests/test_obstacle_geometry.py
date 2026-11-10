import sys,math
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from humanoid_imitation.planning.obstacle_geometry import signed_distance,body_clearance,avoidance_direction,motion_cost


def test_circle_and_rotated_box_exact_distances():
    p=torch.tensor([[[0.,0.],[2.,0.]],[[0.,0.],[2.,0.]]]);center=torch.zeros(2,2);half=torch.tensor([[.6,.6],[1.,.5]]);angle=torch.tensor([0.,math.pi/2]);circle=torch.tensor([True,False])
    assert torch.allclose(signed_distance(p,center,half,angle,circle),torch.tensor([[-.6,1.4],[-.5,1.5]]),atol=1e-6)


def test_segment_crossing_thin_wall_is_not_missed():
    body=torch.tensor([[[-1.,0.,1.],[1.,0.,1.]]]);clear=body_clearance(body,torch.tensor([0,0]),torch.zeros(1,2),torch.tensor([[.02,1.]]),torch.zeros(1),torch.tensor([False]))
    assert clear.item()<0


def test_early_turn_side_is_preserved_and_deactivates_after_passing():
    pos=torch.tensor([[0.,0.],[4.,1.5]]);goal=torch.tensor([[6.,0.],[6.,0.]]);center=torch.tensor([[2.8,0.],[2.8,0.]]);half=torch.ones(2,2)*.6
    direction,active=avoidance_direction(pos,goal,center,half,torch.zeros(2),torch.ones(2,dtype=torch.bool),torch.zeros(2),torch.ones(2))
    assert direction[0]>0 and active.tolist()==[True,False] and direction[1]<0


def test_geometry_cost_gradient_is_finite():
    root=torch.zeros(1,60,71);root[:,:,0]=torch.linspace(.03,2,60);root[:,:,3]=1;root[:,:,9]=1;root.requires_grad_()
    value=motion_cost(root,root.detach(),torch.tensor([[2.8,.1]]),torch.tensor([[.125,1.]]),torch.zeros(1),torch.tensor([False]),anticipate=True)
    grad=torch.autograd.grad(value.sum(),root)[0]
    assert torch.isfinite(value).all() and torch.isfinite(grad).all() and grad.norm()>0


def test_early_curve_velocity_and_decoder_prefix_steering():
    from humanoid_imitation.planning.obstacle_geometry import guidance_target
    root=torch.zeros(1,60,71);root[:,:,0]=torch.arange(1,61)/30;root[:,:,3]=1;root[:,:,7]=1;root[:,:,9]=1
    target=guidance_target(root,torch.zeros(1),torch.tensor([math.pi/6]))
    heading=torch.atan2(target[0,29,4],target[0,29,3])
    assert heading>math.radians(25)
    velocity=torch.diff(torch.cat((torch.zeros(1,1,2),target[:,:,:2]),1),dim=1)*30
    assert torch.allclose(velocity,target[:,:,9:11],atol=1e-6)
    assert torch.allclose(velocity.norm(dim=-1),torch.ones(1,60),atol=1e-5)


def test_box_render_footprint_matches_sdf_geometry():
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
    from motion_video_renderer import obstacle_outline
    import numpy as np
    points=obstacle_outline(dict(kind='box',xy=[2.,3.],half=[.125,1.],angle=math.pi/2))
    assert np.allclose(points.min(0),[1.,2.875])
    assert np.allclose(points.max(0),[3.,3.125])
    assert np.allclose(points[0],points[-1])


def test_explicit_retarget_preserves_height_speed_and_disabled_items():
    from humanoid_imitation.planning.obstacle_root_retarget import retarget_root
    root=torch.zeros(2,60,71);root[:,:,0]=torch.arange(1,61)/30;root[:,:,2]=.9;root[:,:,3]=1;root[:,:,7]=1;root[:,:,9]=1
    target=retarget_root(root,torch.zeros(2),torch.tensor([.7,.7]),torch.tensor([True,False]))
    assert torch.equal(target[1],root[1]) and torch.equal(target[:,:,2],root[:,:,2])
    assert torch.allclose(target[0,:,9:11].norm(dim=-1),torch.ones(60),atol=1e-5)
    assert torch.atan2(target[0,29,4],target[0,29,3])>math.radians(35)


def test_cost_target_does_not_mutate_nominal_and_has_finite_descent_gradient():
    from humanoid_imitation.planning.obstacle_geometry import guidance_target
    root=torch.zeros(1,60,71);root[:,:,0]=torch.arange(1,61)/30;root[:,:,3]=1;root[:,:,7]=1;root[:,:,9]=1
    original=root.clone()
    target=guidance_target(root,torch.zeros(1),torch.tensor([math.pi/3]),turn_cap_deg=40.)
    assert torch.equal(root,original)
    assert math.radians(35)<torch.atan2(target[0,29,4],target[0,29,3])<math.radians(41)
    prediction=root.clone().requires_grad_(True)
    args=(root,torch.tensor([[2.8,0.]]),torch.tensor([[.125,1.]]),torch.zeros(1),torch.tensor([False]))
    value=motion_cost(prediction,*args,target=target,anticipate=True)
    grad=torch.autograd.grad(value.sum(),prediction)[0]
    assert torch.isfinite(grad).all() and grad.norm()>0
    improved=prediction.detach()-1e-4*grad/grad.norm()
    assert motion_cost(improved,*args,target=target,anticipate=True)<value.detach()
