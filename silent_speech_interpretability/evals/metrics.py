"""Evaluation metrics for classification and fusion baselines."""

from __future__ import annotations

import numpy as np


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true == y_pred)) if len(y_true) else 0.0


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> np.ndarray:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.asarray(labels if labels is not None else sorted(set(y_true) | set(y_pred)))
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for true, pred in zip(y_true, y_pred):
        matrix[index[true], index[pred]] += 1
    return matrix


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, labels: np.ndarray | None = None) -> float:
    cm = confusion_matrix(y_true, y_pred, labels)
    f1s = []
    for i in range(cm.shape[0]):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1s.append(2 * precision * recall / (precision + recall + 1e-8))
    return float(np.mean(f1s)) if f1s else 0.0


def per_class_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    result = {}
    for cls in sorted(set(np.asarray(y_true))):
        mask = y_true == cls
        result[str(int(cls))] = float(np.mean(y_pred[mask] == y_true[mask])) if mask.any() else 0.0
    return result


def speaker_level_accuracy(y_true: np.ndarray, y_pred: np.ndarray, user_ids: np.ndarray) -> dict[str, float]:
    result = {}
    for speaker in sorted(set(np.asarray(user_ids))):
        mask = user_ids == speaker
        result[str(int(speaker))] = accuracy(y_true[mask], y_pred[mask])
    return result


def bootstrap_accuracy_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_boot: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if len(y_true) == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0}
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        scores.append(accuracy(y_true[idx], y_pred[idx]))
    alpha = (1 - confidence) / 2
    return {
        "mean": accuracy(y_true, y_pred),
        "low": float(np.quantile(scores, alpha)),
        "high": float(np.quantile(scores, 1 - alpha)),
    }
