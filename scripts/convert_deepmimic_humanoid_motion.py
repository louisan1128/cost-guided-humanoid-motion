#!/usr/bin/env python3
"""Convert an original DeepMimic humanoid3d JSON motion to MimicKit PKL.

The formats use the same semantic joints, but differ in coordinate system,
joint order, rotation representation, and shoulder rest frames. The mapping
was checked against the spinkick clip distributed by both repositories.
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import numpy as np


# DeepMimic is Y-up. MimicKit is Z-up and uses -Y for DeepMimic +Z.
_DEEP_TO_MIMIC = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1e-8):
        raise ValueError("Encountered a zero-length quaternion")
    quat = quat / norm
    w, x, y, z = np.moveaxis(quat, -1, 0)
    return np.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(quat.shape[:-1] + (3, 3))


def _matrix_to_expmap(matrix: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to principal exponential-map vectors."""
    matrix = np.asarray(matrix, dtype=np.float64)
    trace = np.trace(matrix, axis1=-2, axis2=-1)
    cos_angle = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arccos(cos_angle)

    skew = np.stack(
        (
            matrix[..., 2, 1] - matrix[..., 1, 2],
            matrix[..., 0, 2] - matrix[..., 2, 0],
            matrix[..., 1, 0] - matrix[..., 0, 1],
        ),
        axis=-1,
    )
    sin_angle = np.sin(angle)
    scale = np.divide(
        angle,
        2.0 * sin_angle,
        out=np.full_like(angle, 0.5),
        where=np.abs(sin_angle) > 1e-7,
    )
    result = skew * scale[..., None]

    near_pi = np.abs(math.pi - angle) < 1e-5
    if np.any(near_pi):
        selected = matrix[near_pi]
        diag = np.maximum(
            (np.diagonal(selected, axis1=-2, axis2=-1) + 1.0) * 0.5,
            0.0,
        )
        axis = np.sqrt(diag)
        axis[:, 0] = np.copysign(
            axis[:, 0], selected[:, 2, 1] - selected[:, 1, 2]
        )
        axis[:, 1] = np.copysign(
            axis[:, 1], selected[:, 0, 2] - selected[:, 2, 0]
        )
        axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
        axis = np.divide(
            axis, axis_norm, out=np.zeros_like(axis), where=axis_norm > 1e-8
        )
        result[near_pi] = axis * angle[near_pi, None]
    return result


def _convert_quaternion(
    quat: np.ndarray, post_rotation: np.ndarray | None = None
) -> np.ndarray:
    source_rotation = _quat_wxyz_to_matrix(quat)
    converted = _DEEP_TO_MIMIC @ source_rotation @ _DEEP_TO_MIMIC.T
    if post_rotation is not None:
        converted = converted @ post_rotation
    return _matrix_to_expmap(converted)


def _infer_fps(frame_durations: np.ndarray) -> float:
    positive = frame_durations[frame_durations > 1e-8]
    if positive.size == 0:
        raise ValueError("Motion has no positive frame duration")
    dt = float(np.median(positive))
    if not np.allclose(positive, dt, rtol=1e-3, atol=1e-7):
        raise ValueError("MimicKit PKL needs constant-rate frames, but durations vary")
    fps = 1.0 / dt
    rounded = round(fps)
    return float(rounded) if abs(fps - rounded) < 1e-2 else fps


def convert_frames(source_frames: np.ndarray) -> np.ndarray:
    if source_frames.ndim != 2 or source_frames.shape[1] != 44:
        raise ValueError(
            f"Expected DeepMimic frames shaped (N, 44), got {source_frames.shape}"
        )

    source = source_frames[:, 1:]
    output = np.empty((source.shape[0], 34), dtype=np.float64)

    output[:, 0] = source[:, 0]
    output[:, 1] = -source[:, 2]
    output[:, 2] = source[:, 1]
    output[:, 3:6] = _convert_quaternion(source[:, 3:7])

    # MimicKit DFS order: abdomen, neck, right arm, left arm, right leg, left leg.
    output[:, 6:9] = _convert_quaternion(source[:, 7:11])
    output[:, 9:12] = _convert_quaternion(source[:, 11:15])

    # DeepMimic uses a downward arm rest pose; MimicKit extends the arms sideways.
    output[:, 12:15] = _convert_quaternion(source[:, 24:28], _DEEP_TO_MIMIC)
    output[:, 15] = source[:, 28]
    output[:, 16:19] = _convert_quaternion(source[:, 38:42], _DEEP_TO_MIMIC.T)
    output[:, 19] = -source[:, 42]

    output[:, 20:23] = _convert_quaternion(source[:, 15:19])
    output[:, 23] = -source[:, 19]
    output[:, 24:27] = _convert_quaternion(source[:, 20:24])
    output[:, 27:30] = _convert_quaternion(source[:, 29:33])
    output[:, 30] = -source[:, 33]
    output[:, 31:34] = _convert_quaternion(source[:, 34:38])

    if not np.isfinite(output).all():
        raise ValueError("Converted motion contains NaN or infinity")
    return output.astype(np.float32)


def convert_file(input_path: Path, output_path: Path) -> tuple[int, float, float]:
    with input_path.open("r", encoding="utf-8") as stream:
        source = json.load(stream)

    source_frames = np.asarray(source["Frames"], dtype=np.float64)
    fps = _infer_fps(source_frames[:, 0])
    frames = convert_frames(source_frames)
    output = {"loop_mode": 0, "fps": fps, "frames": frames.tolist()}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as stream:
        pickle.dump(output, stream)

    duration = (len(frames) - 1) / fps
    return len(frames), fps, duration


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="DeepMimic humanoid3d JSON/TXT")
    parser.add_argument("output", type=Path, help="Output MimicKit PKL")
    args = parser.parse_args()

    count, fps, duration = convert_file(args.input, args.output)
    print(f"saved: {args.output}")
    print(f"frames: {count}, fps: {fps:g}, duration: {duration:.3f} s, loop: clamp")


if __name__ == "__main__":
    main()
