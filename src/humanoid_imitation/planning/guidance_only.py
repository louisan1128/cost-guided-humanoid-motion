"""Test-time costs and denoising updates without a supplied route or output edits."""
import torch
from .obstacle_geometry import motion_cost, signed_distance
from .obstacle_noise_guidance import cost_terms
from .state_action_sequence import sample_sequence


def task_cost(root, nominal, center, half, angle, circle, goal):
    base = motion_cost(root, nominal, center, half, angle, circle, anticipate=True)
    old = cost_terms(root, nominal, center, radius=0.)['progress']
    direction = goal / goal.norm(dim=-1, keepdim=True).clamp_min(.1)
    # Allow lateral detours; penalize only substantial lost goalward progress.
    nominal_progress = (nominal[:, -1, :2] * direction).sum(-1)
    progress = (root[:, -1, :2] * direction).sum(-1)
    loss = torch.relu(nominal_progress - progress - .4).square()
    distance = (root[..., :2] - goal[:, None]).norm(dim=-1)
    near = torch.exp(-2 * distance.detach())
    stop = (near * root[..., 9:11].square().sum(-1)).mean(1)
    return base - old + loss + .2 * stop


def risk_gate(nominal, center, half, angle, circle):
    distance = signed_distance(nominal[..., :2], center, half, angle, circle)
    times = torch.linspace(.25, 2., 8, device=nominal.device)
    future = nominal[:, -1:, :2] + times[None, :, None] * nominal[:, -1:, 9:11]
    preview = signed_distance(future, center, half, angle, circle)
    return (distance.amin(1) < .8) | (preview.amin(1) < .8)


def guided_sample(model, context, alpha, noise, guide=None, scale=30., variant='legacy', steps=16, diagnostics=None):
    if guide is None or not scale or variant == 'legacy':
        return sample_sequence(model, context, alpha, noise, steps, guide, scale, True)
    if model.method == 'mlp':
        return model(context)
    x = noise
    times = torch.linspace(len(alpha)-1, 0, steps).round().long().tolist()
    for j, t in enumerate(times):
        with torch.enable_grad():
            variable = x.detach().requires_grad_(True)
            clean = model(context, variable, torch.full((len(x),), t, device=x.device, dtype=torch.long))
            value = guide(clean)
            gradient = torch.autograd.grad(value.sum(), variable)[0]
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite guidance gradient')
            norm = gradient.norm(dim=-1, keepdim=True)
            # Bound strong gradients; retain the magnitude of weak signals.
            gradient = gradient / norm.clamp_min(1.)
            # Reduce late updates so the final denoiser can restore consistency.
            strength = 1. if j < steps*2/3 else .25
            delta = scale * strength * (1-alpha[t]).sqrt() * gradient
            x = (variable-delta).detach()
            if diagnostics is not None:
                diagnostics.append(dict(step=j, t=t, mean_cost=float(value.mean()),
                                        mean_gradient_norm=float(norm.mean()), max_update_norm=float(delta.norm(dim=-1).max())))
        with torch.no_grad():
            clean = model(context, x, torch.full((len(x),), t, device=x.device, dtype=torch.long))
            if j == len(times)-1:
                return clean
            a, b = alpha[t], alpha[times[j+1]]
            eps = (x-a.sqrt()*clean)/(1-a).sqrt()
            x = b.sqrt()*clean+(1-b).sqrt()*eps
    raise RuntimeError('Empty denoising schedule')
