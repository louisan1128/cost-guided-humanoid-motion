"""Shared physical action decoder explicitly conditioned on desired root motion.

Future observations are a desired-motion condition, not current-state inputs.
At deployment the motion generator must supply this condition. Oracle futures
are diagnostic only. No future actions or future joints are decoder inputs.
"""
import torch
from torch import nn

INDICES = (2, 7, 14, 29)
CHANNELS = (0, 1, 2, 3, 4, 9, 10, 14)


def root_features(states):
    if states.shape[-2:] != (30, 71):
        raise ValueError('Expected H30 physical states71')
    return states[..., list(INDICES), :][..., list(CHANNELS)].flatten(-2)


def fixed_travel_observation(obs, root_pos, root_yaw, goal_xy, max_distance=1.):
    """Task-only change for travel far outside arrival mode; preserve physical obs."""
    delta = goal_xy - root_pos[..., :2]
    delta = delta * (max_distance / delta.norm(dim=-1, keepdim=True).clamp_min(1e-6)).clamp(max=1.)
    c, s = root_yaw.cos(), root_yaw.sin()
    target = torch.atan2(goal_xy[..., 1]-root_pos[..., 1], goal_xy[..., 0]-root_pos[..., 0])
    error = target-root_yaw
    goal = torch.stack((c*delta[..., 0]+s*delta[..., 1], -s*delta[..., 0]+c*delta[..., 1], error.sin(), error.cos()), -1)
    return torch.cat((obs[..., :-4], goal), -1)


class FutureConditionedAction(nn.Module):
    def __init__(self, prior, action_std, feature_mean, feature_std, limit_rad=.5, include_goal=False):
        super().__init__()
        self.prior = prior.eval().requires_grad_(False)
        self.include_goal = include_goal
        self.residual = nn.Sequential(nn.Linear(176 if include_goal else 172,512),nn.SiLU(),nn.Linear(512,256),nn.SiLU(),nn.Linear(256,28))
        self.register_buffer('feature_mean', feature_mean.clone())
        self.register_buffer('feature_std', feature_std.clone())
        self.register_buffer('action_std', action_std.clone())
        self.limit_rad = limit_rad
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def forward(self, normalized_obs, features, noise=0.):
        feature = ((features-self.feature_mean)/self.feature_std).clamp(-10,10)
        if noise:
            feature = feature + torch.randn_like(feature)*noise
        current = normalized_obs if self.include_goal else normalized_obs[..., :140]
        raw = self.residual(torch.cat((current, feature), -1))
        correction = self.limit_rad * torch.tanh(raw*self.action_std/self.limit_rad)
        return self.prior(normalized_obs) + correction/self.action_std
