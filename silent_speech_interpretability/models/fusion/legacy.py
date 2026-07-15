"""Original fusionGate-style prototype fusion helpers."""

from __future__ import annotations

import numpy as np


def compute_stacked_prototypes(x: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    num_modalities = x.shape[1]
    embedding_dim = x.shape[2]
    prototypes = np.zeros((num_modalities, num_classes, embedding_dim), dtype=np.float32)
    for class_id in range(num_classes):
        mask = y == class_id
        if mask.any():
            prototypes[:, class_id, :] = x[mask].mean(axis=0)
    norms = np.linalg.norm(prototypes, axis=2, keepdims=True).clip(1e-8)
    return prototypes / norms


def legacy_proto_scores(x: np.ndarray, prototypes: np.ndarray, temp: float = 10.0) -> np.ndarray:
    x_norm = x / np.linalg.norm(x, axis=2, keepdims=True).clip(1e-8)
    scores = np.einsum("nkd,kcd->nkc", x_norm, prototypes) * temp
    scores -= scores.max(axis=2, keepdims=True)
    exp_scores = np.exp(scores)
    return exp_scores / exp_scores.sum(axis=2, keepdims=True)


def legacy_equal_weight_predictions(x: np.ndarray, prototypes: np.ndarray, temp: float = 10.0) -> np.ndarray:
    scores = legacy_proto_scores(x, prototypes, temp)
    return scores.mean(axis=1).argmax(axis=1)


def legacy_borda_predictions(x: np.ndarray, prototypes: np.ndarray, temp: float = 10.0) -> np.ndarray:
    scores = legacy_proto_scores(x, prototypes, temp)
    ranks = np.argsort(np.argsort(-scores, axis=2), axis=2)
    borda = ranks.sum(axis=1)
    return borda.argmin(axis=1)


def legacy_hard_consistency_predictions(x: np.ndarray, prototypes: np.ndarray, temp: float = 10.0) -> np.ndarray:
    scores = legacy_proto_scores(x, prototypes, temp)
    preds = scores.argmax(axis=2)
    n_samples, n_modalities = preds.shape
    agreement = np.zeros((n_samples, n_modalities), dtype=np.float32)
    for modality_idx in range(n_modalities):
        for other_idx in range(n_modalities):
            if other_idx != modality_idx:
                agreement[:, modality_idx] += (preds[:, modality_idx] == preds[:, other_idx]).astype(np.float32)
    agreement = agreement / (n_modalities - 1) + 0.1
    agreement = agreement / agreement.sum(axis=1, keepdims=True).clip(1e-8)
    return (scores * agreement[:, :, None]).sum(axis=1).argmax(axis=1)
