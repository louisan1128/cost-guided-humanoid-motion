"""Supervised joint sequence correction around a frozen direct-action prior.

The prior remains a learned MLP/diffusion action policy; this is not an RL
residual controller or a stop/reference-selection rule.
"""
import torch
from torch import nn

from .short_action_sequence import ShortActionSequenceNet


class AnchoredActionSequenceNet(nn.Module):
    def __init__(self, prior, method, action_std, horizon=30, action_horizon=8,
                 correction_limit_rad=.03, width=512):
        super().__init__()
        if correction_limit_rad <= 0:
            raise ValueError('Correction limit must be positive')
        self.method = method
        self.prior = prior.eval().requires_grad_(False)
        self.future = ShortActionSequenceNet(660, horizon, action_horizon, method, width)
        self.register_buffer('correction_limit', correction_limit_rad / action_std.clamp_min(1e-4))
        with torch.no_grad():
            self.future.output[-1].weight[:28].zero_()
            self.future.output[-1].bias[:28].zero_()

    @property
    def valid_output(self):
        return self.future.valid_output

    def forward(self, context, noisy=None, t=None):
        current_obs = context[:, 3*144:4*144]
        base = self.prior(current_obs) if self.method == 'mlp' else self.prior(current_obs, noisy[:, :28], t)
        joint = self.future(context, noisy, t)
        correction = self.correction_limit * torch.tanh(joint[:, :28] / self.correction_limit)
        return torch.cat((base + correction, joint[:, 28:]), -1)
