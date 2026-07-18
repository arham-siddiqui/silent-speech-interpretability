"""Relative-time pooling and lip-articulation targets."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def pool_temporal_segments(
    sequences: torch.Tensor,
    lengths: torch.Tensor,
    num_segments: int,
) -> torch.Tensor:
    """Mean-pool each valid sequence into an equal number of relative-time regions."""
    if sequences.ndim != 3:
        raise ValueError("sequences must have shape [batch, time, features]")
    if num_segments < 1:
        raise ValueError("num_segments must be positive")
    pooled = []
    for sequence, length in zip(sequences, lengths, strict=True):
        valid = sequence[: int(length)]
        if len(valid) < num_segments:
            raise ValueError("every sequence must contain at least num_segments valid steps")
        boundaries = torch.linspace(0, len(valid), num_segments + 1, device=valid.device).round().long()
        sample = torch.stack([valid[boundaries[i] : boundaries[i + 1]].mean(dim=0) for i in range(num_segments)])
        pooled.append(F.normalize(sample, p=2, dim=-1))
    return torch.stack(pooled)


def lip_articulation_segments(lip_sequence: np.ndarray, num_segments: int) -> np.ndarray:
    """Return aperture, width, and movement trajectories from normalized lip landmarks."""
    sequence = np.asarray(lip_sequence, dtype=np.float32)
    if sequence.ndim != 2 or sequence.shape[1] != 40:
        raise ValueError("lip_sequence must contain 20 flattened 2D lip landmarks")
    points = sequence.reshape(len(sequence), 20, 2)
    aperture = np.linalg.norm(points[:, 14] - points[:, 18], axis=1)
    width = np.linalg.norm(points[:, 0] - points[:, 6], axis=1)
    movement = np.linalg.norm(np.gradient(points, axis=0), axis=2).mean(axis=1)
    trajectory = np.stack([aperture, width, movement], axis=1)
    if len(trajectory) < num_segments:
        raise ValueError("lip sequence is shorter than num_segments")
    return np.stack([part.mean(axis=0) for part in np.array_split(trajectory, num_segments)]).astype(np.float32)
