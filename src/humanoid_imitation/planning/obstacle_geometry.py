"""Known planar geometry and anticipatory motion costs. No learned perception."""
import math
import torch
from .state_action_sequence import rotate_z
from .obstacle_noise_guidance import cost_terms
from .consistent_root_guidance import curved_root_target,root_path_cost


def signed_distance(points,center,half,angle,circle):
    """Batched circle/OBB signed distance; arbitrary point axes after batch."""
    shape=(len(points),)+(1,)*(points.ndim-2)
    delta=points-center.reshape(*shape,2)
    a=angle.reshape(shape);x=a.cos()*delta[...,0]+a.sin()*delta[...,1];y=-a.sin()*delta[...,0]+a.cos()*delta[...,1]
    q=torch.stack((x.abs(),y.abs()),-1)-half.reshape(*shape,2)
    box=q.clamp_min(0).norm(dim=-1)+q.amax(-1).clamp_max(0)
    disk=delta.norm(dim=-1)-half[:,0].reshape(shape)
    return torch.where(circle.reshape(shape),disk,box)


def body_clearance(body,parents,center,half,angle,circle,height=1.5):
    # 33 samples plus Lipschitz spacing correction lower-bounds segment SDF.
    start=body[:,parents];delta=body-start
    t=torch.linspace(0,1,33,device=body.device)
    samples=start[:,:,None]+t[None,None,:,None]*delta[:,:,None]
    dist=signed_distance(samples[...,:2],center,half,angle,circle)
    dist=torch.where(samples[...,2]<height+.08,dist,torch.full_like(dist,float('inf')))
    return (dist.amin(-1)-delta[:,:,:2].norm(dim=-1)/64).amin(-1)


def avoidance_direction(position,goal,center,half,angle,circle,axis,side,margin=.7):
    """Choose one side once, then turn early until past the obstacle's far face.

    This is an analytic geometry-based guidance prior, not a generated waypoint.
    side is latched from the initial goal-line/obstacle configuration.
    """
    forward=torch.stack((axis.cos(),axis.sin()),-1);left=torch.stack((-axis.sin(),axis.cos()),-1)
    delta=center-position;ahead=(delta*forward).sum(-1);lateral=((position-center)*left).sum(-1)
    rel=angle-axis
    depth=half[:,0]*rel.cos().abs()+half[:,1]*rel.sin().abs()
    width=half[:,0]*rel.sin().abs()+half[:,1]*rel.cos().abs()
    depth=torch.where(circle,half[:,0],depth);width=torch.where(circle,half[:,0],width)
    needed=(width+margin-side*lateral).clamp_min(0)
    turn=torch.atan2(needed,(ahead-depth).clamp_min(.6)).clamp(max=math.radians(40))
    active=(ahead>-depth-.6)&(ahead<6.)
    direct=torch.atan2(goal[:,1]-position[:,1],goal[:,0]-position[:,0])
    desired=torch.where(active,axis+side*turn,direct)
    return desired,active


def motion_cost(root,nominal,center,half,angle,circle,target=None,anticipate=False):
    distance=signed_distance(root[...,:2],center,half,angle,circle)
    risk=(torch.relu(.5-distance)/.2).square()
    collision=risk.topk(min(5,root.shape[1]),dim=1).values.mean(1)
    terms=cost_terms(root,nominal,center,radius=0.)
    cost=10*collision+terms['consistency']+.5*terms['course']+terms['progress']
    if anticipate:
        # Constant-terminal-velocity preview is a cost approximation, not rollout.
        times=torch.linspace(.25,2.,8,device=root.device)
        future=root[:,-1:,:2]+times[None,:,None]*root[:,-1:,9:11]
        preview=signed_distance(future,center,half,angle,circle)
        cost=cost+3*(torch.relu(.6-preview)/.3).square().amax(-1)
    if target is not None:cost=cost+3*root_path_cost(root,target)
    return cost


def guidance_target(nominal,anchor,desired,turn_cap_deg=30.,ramp_s=1.2):
    # Decoder consumes only the first second of H60; steering must occur there.
    root=nominal.detach().clone();n=root.shape[1]
    direction=torch.atan2(root[:,-1,1],root[:,-1,0])
    cap=math.radians(turn_cap_deg)
    delta=desired-anchor-direction;delta=torch.atan2(delta.sin(),delta.cos()).clamp(-cap,cap)
    phase=(torch.arange(1,n+1,device=root.device,dtype=root.dtype)/30/ramp_s).clamp(max=1)
    ramp=phase.square()*(3-2*phase);angle=delta[:,None]*ramp
    increments=torch.diff(torch.cat((torch.zeros_like(root[:,:1,:3]),root[...,:3]),1),dim=1)
    root[...,:3]=rotate_z(increments,angle).cumsum(1)
    root[...,3:6]=rotate_z(root[...,3:6],angle);root[...,6:9]=rotate_z(root[...,6:9],angle)
    root[...,9:11]=torch.diff(torch.cat((torch.zeros_like(root[:,:1,:2]),root[...,:2]),1),dim=1)*30
    heading=torch.atan2(root[...,4],root[...,3]);dh=torch.diff(torch.cat((torch.zeros_like(heading[:,:1]),heading),1),dim=1)
    root[...,14]=torch.atan2(dh.sin(),dh.cos())*30
    return root
