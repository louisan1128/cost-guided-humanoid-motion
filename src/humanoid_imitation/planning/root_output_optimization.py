"""Inference-only optimization of one predicted root path, shared by MLP/DP."""
import torch
from .state_action_sequence import rotate_z


def parameterized_root(nominal, perturbation, dt=1/30):
    """XY units .2m, yaw units .3rad; preserve z and rotate both 6D columns."""
    xy = nominal[..., :2] + .2 * perturbation[..., :2]
    angle = .3 * perturbation[..., 2]
    c1 = rotate_z(nominal[..., 3:6], angle)
    c2 = rotate_z(nominal[..., 6:9], angle)
    velocity = torch.diff(torch.cat((torch.zeros_like(xy[:, :1]), xy), 1), dim=1)/dt
    heading = torch.atan2(c1[..., 1], c1[..., 0])
    difference = torch.diff(torch.cat((torch.zeros_like(heading[:, :1]), heading), 1), dim=1)
    angular = torch.atan2(difference.sin(), difference.cos())/dt
    return torch.cat((xy, nominal[..., 2:3], c1, c2, velocity,
                      nominal[..., 11:14], angular[..., None], nominal[..., 15:]), -1)


def optimize_root(nominal, cost, active, lr=.03, steps=32):
    """One path/Adam run, retain lowest-cost iterate; never update model weights.

    Disabled rows are exactly unchanged. The original nominal is also retained
    if coordinate-consistent reconstruction/optimization does not improve cost.
    This is output optimization, not denoising guidance or K-sample search.
    """
    nominal = nominal.detach()
    if not bool(active.any()):
        return nominal
    with torch.enable_grad():
        variable = torch.zeros(*nominal.shape[:2], 3, device=nominal.device,
                               dtype=nominal.dtype, requires_grad=True)
        optimizer = torch.optim.Adam([variable], lr=lr)
        with torch.no_grad():
            best = nominal.clone()
            best_cost = cost(best).detach()
        for _ in range(steps):
            predicted = parameterized_root(nominal, variable)
            value = cost(predicted)
            gradient = torch.autograd.grad((value*active).sum(), variable)[0]
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite output-optimization gradient')
            optimizer.zero_grad()
            variable.grad = gradient
            optimizer.step()
            with torch.no_grad():
                candidate = parameterized_root(nominal, variable)
                candidate_cost = cost(candidate)
                if not torch.isfinite(candidate_cost).all():
                    raise ValueError('Nonfinite output cost')
                improved = active & (candidate_cost < best_cost)
                best = torch.where(improved[:, None, None], candidate, best)
                best_cost = torch.where(improved, candidate_cost, best_cost)
    return best.detach()
