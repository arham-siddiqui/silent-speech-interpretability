"""Selectivity and cross-fold stability metrics for sparse features."""

from __future__ import annotations

import numpy as np


def eta_squared(features: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """One-way ANOVA effect size for every feature."""
    features = np.asarray(features, dtype=np.float64)
    labels = np.asarray(labels)
    grand_mean = features.mean(axis=0)
    between = np.zeros(features.shape[1], dtype=np.float64)
    for label in np.unique(labels):
        group = features[labels == label]
        between += len(group) * (group.mean(axis=0) - grand_mean) ** 2
    total = ((features - grand_mean) ** 2).sum(axis=0)
    return np.divide(between, total, out=np.zeros_like(between), where=total > 1e-12).astype(np.float32)


def feature_rankings(
    features: np.ndarray,
    class_labels: np.ndarray,
    type_labels: np.ndarray,
    speaker_labels: np.ndarray,
) -> dict[str, np.ndarray]:
    class_selectivity = eta_squared(features, class_labels)
    type_selectivity = eta_squared(features, type_labels)
    speaker_selectivity = eta_squared(features, speaker_labels)
    activation_frequency = (features > 1e-6).mean(axis=0).astype(np.float32)
    raw_content_score = class_selectivity + 0.5 * type_selectivity - 0.5 * speaker_selectivity
    frequency_weight = np.sqrt(4.0 * activation_frequency * (1.0 - activation_frequency))
    content_score = raw_content_score * frequency_weight
    valid = (activation_frequency >= 0.01) & (activation_frequency <= 0.95)
    content_score = np.where(valid, content_score, -np.inf)
    return {
        "class_selectivity": class_selectivity,
        "type_selectivity": type_selectivity,
        "speaker_selectivity": speaker_selectivity,
        "activation_frequency": activation_frequency,
        "mean_activation": features.mean(axis=0).astype(np.float32),
        "content_score": content_score.astype(np.float32),
        "frequency_weight": frequency_weight.astype(np.float32),
        "valid": valid,
        "rank": np.argsort(-content_score),
    }


def decoder_stability(decoder_vector: np.ndarray, other_decoder_matrices: list[np.ndarray]) -> float:
    """Mean best positive cosine match in independently trained folds."""
    vector = decoder_vector / (np.linalg.norm(decoder_vector) + 1e-8)
    matches = []
    for matrix in other_decoder_matrices:
        normalized = matrix / (np.linalg.norm(matrix, axis=0, keepdims=True) + 1e-8)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            matches.append(float(np.max(vector @ normalized)))
    return float(np.mean(matches)) if matches else 1.0
