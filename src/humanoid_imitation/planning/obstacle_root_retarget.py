"""Explicit analytic root-motion correction; distinct from denoising-only guidance."""
import math
import torch
from .state_action_sequence import rotate_z


def retarget_root(nominal,anchor,desired,active,turn_cap_deg=40.,ramp_s=1.2):
    root=nominal.detach().clone();n=root.shape[1]
    direction=torch.atan2(root[:,-1,1],root[:,-1,0]);delta=desired-anchor-direction
    delta=torch.atan2(delta.sin(),delta.cos()).clamp(-math.radians(turn_cap_deg),math.radians(turn_cap_deg))
    phase=(torch.arange(1,n+1,device=root.device,dtype=root.dtype)/30/ramp_s).clamp(max=1)
    angle=delta[:,None]*phase.square()*(3-2*phase)
    increments=torch.diff(torch.cat((torch.zeros_like(root[:,:1,:3]),root[...,:3]),1),dim=1)
    root[...,:3]=rotate_z(increments,angle).cumsum(1)
    for start in [3,6]:root[...,start:start+3]=rotate_z(root[...,start:start+3],angle)
    root[...,9:11]=torch.diff(torch.cat((torch.zeros_like(root[:,:1,:2]),root[...,:2]),1),dim=1)*30
    heading=torch.atan2(root[...,4],root[...,3]);dh=torch.diff(torch.cat((torch.zeros_like(heading[:,:1]),heading),1),dim=1)
    root[...,14]=torch.atan2(dh.sin(),dh.cos())*30
    return torch.where(active[:,None,None],root,nominal)
