"""No-training fusion rules for modality probability vectors."""

from __future__ import annotations

import numpy as np


def equal_weight_fusion(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    stacked = np.stack(list(probabilities.values()), axis=0)
    return stacked.mean(axis=0)


def static_weight_fusion(probabilities: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    """Fuse probability vectors with fixed modality weights."""
    modalities = list(probabilities)
    raw_weights = np.asarray([weights.get(modality, 0.0) for modality in modalities], dtype=np.float32)
    raw_weights = np.maximum(raw_weights, 0.0)
    if float(raw_weights.sum()) <= 0.0:
        raw_weights = np.ones(len(modalities), dtype=np.float32)
    normalized = raw_weights / (raw_weights.sum() + 1e-8)
    stacked = np.stack([probabilities[modality] for modality in modalities], axis=0)
    return np.sum(stacked * normalized[:, None, None], axis=0)


def borda_count_fusion(probabilities: dict[str, np.ndarray]) -> np.ndarray:
    prob_list = list(probabilities.values())
    n_samples, n_classes = prob_list[0].shape
    scores = np.zeros((n_samples, n_classes), dtype=np.float32)
    for probs in prob_list:
        order = np.argsort(-probs, axis=1)
        ranks = np.empty_like(order)
        ranks[np.arange(n_samples)[:, None], order] = np.arange(n_classes)
        scores += n_classes - ranks
    return scores / (scores.sum(axis=1, keepdims=True) + 1e-8)


def consistency_weighted_fusion(probabilities: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    modalities = list(probabilities)
    stacked = np.stack([probabilities[m] for m in modalities], axis=1)
    n_samples, n_modalities, _ = stacked.shape
    weights = np.ones((n_samples, n_modalities), dtype=np.float32)

    for i in range(n_modalities):
        agreements = []
        for j in range(n_modalities):
            if i == j:
                continue
            dot = np.sum(stacked[:, i, :] * stacked[:, j, :], axis=1)
            left = np.linalg.norm(stacked[:, i, :], axis=1)
            right = np.linalg.norm(stacked[:, j, :], axis=1)
            agreements.append(dot / (left * right + 1e-8))
        weights[:, i] = np.mean(np.stack(agreements, axis=1), axis=1)

    weights = np.maximum(weights, 1e-6)
    weights = weights / weights.sum(axis=1, keepdims=True)
    fused = np.sum(stacked * weights[:, :, None], axis=1)
    return fused, weights
