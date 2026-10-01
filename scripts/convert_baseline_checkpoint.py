"""Expand a DeepMimic checkpoint for one additional speed-command input."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from pathlib import Path

import torch


OBS_MEAN_KEY = "_obs_norm._mean"
OBS_STD_KEY = "_obs_norm._std"
INPUT_WEIGHT_KEYS = (
    "_model._actor_layers.0.weight",
    "_model._critic_layers.0.weight",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--command-mean", type=float, default=1.0)
    parser.add_argument("--command-std", type=float, default=0.2)
    return parser.parse_args()


def append_scalar(tensor: torch.Tensor, value: float) -> torch.Tensor:
    scalar = tensor.new_tensor([value])
    return torch.cat((tensor, scalar), dim=0)


def append_zero_input(weight: torch.Tensor) -> torch.Tensor:
    zero_column = weight.new_zeros((weight.shape[0], 1))
    return torch.cat((weight, zero_column), dim=1)


def convert_checkpoint(
    state: Mapping[str, object], command_mean: float, command_std: float
) -> dict[str, object]:
    if command_std <= 0.0:
        raise ValueError("--command-std must be positive")

    required = (OBS_MEAN_KEY, OBS_STD_KEY, *INPUT_WEIGHT_KEYS)
    missing = [key for key in required if key not in state]
    if missing:
        raise KeyError(f"Checkpoint is missing required tensors: {missing}")

    converted = dict(state)
    obs_mean = state[OBS_MEAN_KEY]
    obs_std = state[OBS_STD_KEY]
    if not isinstance(obs_mean, torch.Tensor) or not isinstance(obs_std, torch.Tensor):
        raise TypeError("Observation normalizer entries must be tensors")
    if obs_mean.ndim != 1 or obs_std.shape != obs_mean.shape:
        raise ValueError("Unexpected observation normalizer shapes")

    old_obs_dim = obs_mean.shape[0]
    converted[OBS_MEAN_KEY] = append_scalar(obs_mean, command_mean)
    converted[OBS_STD_KEY] = append_scalar(obs_std, command_std)

    for key in INPUT_WEIGHT_KEYS:
        weight = state[key]
        if not isinstance(weight, torch.Tensor) or weight.ndim != 2:
            raise TypeError(f"{key} must be a rank-2 tensor")
        if weight.shape[1] != old_obs_dim:
            raise ValueError(
                f"{key} input size {weight.shape[1]} does not match obs size {old_obs_dim}"
            )
        converted[key] = append_zero_input(weight)

    return converted


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing file: {output_path}")

    state = torch.load(input_path, map_location="cpu")
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected a state dictionary, got {type(state).__name__}")

    converted = convert_checkpoint(state, args.command_mean, args.command_std)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(converted, output_path)

    old_dim = state[OBS_MEAN_KEY].shape[0]
    new_dim = converted[OBS_MEAN_KEY].shape[0]
    print(f"Converted observation size: {old_dim} -> {new_dim}")
    print(f"Command normalization: mean={args.command_mean}, std={args.command_std}")
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
