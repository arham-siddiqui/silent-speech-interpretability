"""Sparse-feature ablation utilities."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from silent_speech_interpretability.interp.sae import SparseAutoencoder
from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent


@torch.no_grad()
def reconstructed_bottleneck(
    sae: SparseAutoencoder,
    bottleneck: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    feature_indices: np.ndarray | list[int] | None = None,
    replacement: np.ndarray | float = 0.0,
    device: torch.device | str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    normalized = torch.tensor((bottleneck - mean) / std, dtype=torch.float32, device=device)
    features = sae.encode(normalized)
    if feature_indices is not None and len(feature_indices):
        indices = torch.tensor(np.asarray(feature_indices), dtype=torch.long, device=device)
        if np.isscalar(replacement):
            features[:, indices] = float(replacement)
        else:
            values = torch.tensor(np.asarray(replacement)[np.asarray(feature_indices)], dtype=torch.float32, device=device)
            features[:, indices] = values
    decoded = sae.decode(features).cpu().numpy().astype(np.float32)
    return decoded * std + mean, features.cpu().numpy().astype(np.float32)


@torch.no_grad()
def evaluate_bottleneck(
    student: ArticulatoryStudent,
    bottleneck: np.ndarray,
    labels: np.ndarray,
    teacher_targets: np.ndarray,
    device: torch.device | str = "cpu",
) -> dict[str, float]:
    values = torch.tensor(bottleneck, dtype=torch.float32, device=device)
    targets = F.normalize(torch.tensor(teacher_targets, dtype=torch.float32, device=device), p=2, dim=-1)
    predicted_targets = F.normalize(student.target_head(values), p=2, dim=-1)
    predictions = student.classifier(values).argmax(dim=-1).cpu().numpy()
    return {
        "accuracy": float(np.mean(predictions == labels)),
        "target_cosine": float(F.cosine_similarity(predicted_targets, targets, dim=-1).mean().item()),
        "target_mse": float(((predicted_targets - targets) ** 2).sum(dim=-1).mean().item()),
    }
