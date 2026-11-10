"""Differentiable costs for a curved root prediction; no waypoint observation override."""
import torch
from .state_action_sequence import rotate_z


def curved_root_target(nominal, anchor_yaw, desired_world_direction, dt=1/30):
    """Curve the nominal planar path smoothly toward a fixed world direction.

    The target is a cost reference, never directly executed. Velocities are
    recomputed from positions and yaw rate from heading, in the same local frame.
    """
    root = nominal.detach().clone()
    n = root.shape[-2]
    direction = torch.atan2(root[:, -1, 1], root[:, -1, 0])
    delta = desired_world_direction - anchor_yaw - direction
    delta = torch.atan2(delta.sin(), delta.cos()).clamp(-torch.pi/6, torch.pi/6)
    ramp = (torch.arange(1, n+1, device=root.device, dtype=root.dtype)/n).square()
    angle = delta[:, None] * ramp
    root[..., :3] = rotate_z(root[..., :3], angle)
    root[..., 3:6] = rotate_z(root[..., 3:6], angle)
    root[..., 6:9] = rotate_z(root[..., 6:9], angle)
    origin = torch.zeros_like(root[:, :1, :2])
    root[..., 9:11] = torch.diff(torch.cat((origin, root[..., :2]), 1), dim=1)/dt
    heading = torch.atan2(root[..., 4], root[..., 3])
    dh = torch.diff(torch.cat((torch.zeros_like(heading[:, :1]), heading), 1), dim=1)
    root[..., 14] = torch.atan2(dh.sin(), dh.cos())/dt
    return root


def root_path_cost(root, target):
    """Physical units: XY .2m, velocity .5m/s, heading .3rad, yaw rate 1rad/s."""
    xy = ((root[..., :2]-target[..., :2])/.2).square().mean((1, 2))
    velocity = ((root[..., 9:11]-target[..., 9:11])/.5).square().mean((1, 2))
    angle = torch.atan2(root[..., 4], root[..., 3]) - torch.atan2(target[..., 4], target[..., 3])
    heading = (2*(1-angle.cos())/.3**2).mean(1)
    angular = (root[..., 14]-target[..., 14]).square().mean(1)
    return xy + velocity + heading + .1*angular
