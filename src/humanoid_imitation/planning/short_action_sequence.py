"""Long state / short action ablation, with no unsupervised action-token input."""
import torch

from .state_action_sequence import SequenceNet


class ShortActionSequenceNet(SequenceNet):
    def __init__(self, context_dim, horizon, action_horizon, method, width=512):
        if not 1 <= action_horizon <= horizon:
            raise ValueError("Require 1 <= action horizon <= state horizon")
        super().__init__(context_dim, horizon * 99, method, width)
        self.horizon = horizon
        self.action_horizon = action_horizon
        mask = torch.ones(horizon, 99)
        mask[action_horizon:, :28] = 0
        self.register_buffer("valid_output", mask.flatten())

    def forward(self, context, noisy=None, t=None):
        if noisy is not None:
            noisy = noisy * self.valid_output
        return super().forward(context, noisy, t) * self.valid_output


def reconstruction_losses(prediction, target, action_horizon, first_fraction):
    """Keep action0's share constant when the supervised action horizon changes."""
    if not 0 < first_fraction <= 1:
        raise ValueError("Invalid first-action fraction")
    if not 1 <= action_horizon <= target.shape[1]:
        raise ValueError("Invalid action horizon")
    error = (prediction - target).square()
    action = error[:, :action_horizon, :28].mean(-1)
    if action_horizon == 1:
        action_loss = action[:, 0].mean()
    else:
        action_loss = (
            first_fraction * action[:, 0]
            + (1 - first_fraction) * action[:, 1:].mean(-1)
        ).mean()
    return action_loss, error[:, :, 28:].mean()
