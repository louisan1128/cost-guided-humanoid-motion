"""Small state/action transformer for a bounded architecture diagnostic.

State tokens see states; action i sees states <=i and actions <=i.
State tokens therefore cannot copy noisy future action labels. Noncausal
state-to-state attention carries look-ahead information toward near actions.
"""
import math
import torch
from torch import nn


class TemporalActionSequenceNet(nn.Module):
    def __init__(self, context_dim, horizon, action_horizon, method, width=128):
        super().__init__()
        if not 1 <= action_horizon <= horizon:
            raise ValueError("Invalid action horizon")
        self.method, self.horizon, self.action_horizon = method, horizon, action_horizon
        self.context = nn.Sequential(nn.Linear(context_dim, 512), nn.SiLU(), nn.Linear(512, width))
        self.state_in, self.action_in = nn.Linear(71, width), nn.Linear(28, width)
        self.position = nn.Parameter(torch.randn(1, horizon+action_horizon, width)*.01)
        self.noise_embedding = nn.Sequential(nn.Linear(32, width), nn.SiLU(), nn.Linear(width, width))
        layer = nn.TransformerEncoderLayer(width, 4, width*4, dropout=0., activation='gelu', batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, 2, norm=nn.LayerNorm(width), enable_nested_tensor=False)
        self.state_out, self.action_out = nn.Linear(width, 71), nn.Linear(width, 28)
        mask = torch.ones(horizon+action_horizon, horizon+action_horizon, dtype=torch.bool)
        mask[:horizon, :horizon] = False
        for i in range(action_horizon):
            mask[horizon+i, :i+1] = False
            mask[horizon+i, horizon:horizon+i+1] = False
        self.register_buffer('attention_mask', mask)
        self.register_buffer('frequencies', torch.exp(-math.log(10000)*torch.arange(16)/15))
        valid = torch.ones(horizon, 99)
        valid[action_horizon:, :28] = 0
        self.register_buffer('valid_output', valid.flatten())

    def forward(self, context, noisy=None, t=None):
        batch = len(context)
        if noisy is None:
            sequence = context.new_zeros(batch, self.horizon, 99)
        else:
            sequence = (noisy*self.valid_output).reshape(batch, self.horizon, 99)
        tokens = torch.cat((self.state_in(sequence[:, :, 28:]), self.action_in(sequence[:, :self.action_horizon, :28])), 1)
        tokens = tokens + self.position + self.context(context)[:, None]
        if t is not None:
            levels = t[:, None].expand(-1, tokens.shape[1]) if t.ndim == 1 else t
            if levels.shape != tokens.shape[:2]:
                raise ValueError('Noise levels must match state/action tokens')
            angles = levels[:, :, None].float()*self.frequencies
            tokens = tokens + self.noise_embedding(torch.cat((angles.sin(), angles.cos()), -1))
        tokens = self.transformer(tokens, mask=self.attention_mask)
        states = self.state_out(tokens[:, :self.horizon])
        actions = self.action_out(tokens[:, self.horizon:])
        actions = torch.cat((actions, actions.new_zeros(batch, self.horizon-self.action_horizon, 28)), 1)
        return torch.cat((actions, states), -1).flatten(1)


def independent_noise(clean, noise, levels, alpha, action_horizon):
    h = clean.shape[-1]//99
    seq, eps = clean.reshape(-1, h, 99), noise.reshape(-1, h, 99)
    if levels.shape != (len(seq), h+action_horizon):
        raise ValueError('Unaligned per-token noise')
    states_alpha = alpha[levels[:, :h]][:, :, None]
    actions_alpha = alpha[levels[:, h:]][:, :, None]
    states = states_alpha.sqrt()*seq[:, :, 28:] + (1-states_alpha).sqrt()*eps[:, :, 28:]
    actions = actions_alpha.sqrt()*seq[:, :action_horizon, :28] + (1-actions_alpha).sqrt()*eps[:, :action_horizon, :28]
    actions = torch.cat((actions, actions.new_zeros(len(seq), h-action_horizon, 28)), 1)
    return torch.cat((actions, states), -1).flatten(1)
