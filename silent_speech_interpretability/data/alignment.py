"""Temporal alignment utilities for sensor and teacher sequences."""

from __future__ import annotations

import numpy as np


def resample_sequence_to_times(sequence: np.ndarray, old_times: np.ndarray, new_times: np.ndarray) -> np.ndarray:
    sequence = np.asarray(sequence)
    old_times = np.asarray(old_times)
    new_times = np.asarray(new_times)
    if sequence.ndim == 1:
        return np.interp(new_times, old_times, sequence)
    columns = [np.interp(new_times, old_times, sequence[:, i]) for i in range(sequence.shape[1])]
    return np.stack(columns, axis=1)


def pool_sequence_over_segments(
    sequence: np.ndarray,
    times: np.ndarray,
    segments: list[dict[str, float]],
    mode: str = "mean",
) -> np.ndarray:
    pooled = []
    for segment in segments:
        mask = (times >= segment["start"]) & (times <= segment["end"])
        values = sequence[mask]
        if len(values) == 0:
            pooled.append(np.zeros(sequence.shape[1:], dtype=sequence.dtype))
        elif mode == "mean":
            pooled.append(values.mean(axis=0))
        elif mode == "max":
            pooled.append(values.max(axis=0))
        else:
            raise ValueError(f"Unsupported pooling mode: {mode}")
    return np.stack(pooled, axis=0)


def align_teacher_to_sensor(sensor_times: np.ndarray, teacher_times: np.ndarray, teacher_values: np.ndarray) -> np.ndarray:
    return resample_sequence_to_times(teacher_values, teacher_times, sensor_times)
