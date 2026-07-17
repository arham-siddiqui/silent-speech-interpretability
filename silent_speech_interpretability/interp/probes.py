"""Deterministic linear probes for frozen representations."""

from __future__ import annotations

import numpy as np

from silent_speech_interpretability.evals.metrics import accuracy as classification_accuracy
from silent_speech_interpretability.evals.metrics import macro_f1


def fit_linear_probe(
    train_x: np.ndarray,
    train_y: np.ndarray,
    val_x: np.ndarray,
    val_y: np.ndarray,
    test_x: np.ndarray,
    test_y: np.ndarray,
    *,
    seed: int = 42,
    c_values: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
) -> dict[str, float | int]:
    """Select regularization on validation data, then evaluate a refit probe."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    train_x = np.asarray(train_x, dtype=np.float64)
    val_x = np.asarray(val_x, dtype=np.float64)
    test_x = np.asarray(test_x, dtype=np.float64)

    best_c = c_values[0]
    best_accuracy = -1.0
    for c_value in c_values:
        probe = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=c_value, max_iter=2000, random_state=seed),
        )
        # macOS Accelerate can emit false-positive floating-point warnings for
        # finite BLAS matmuls; fitted coefficients are checked explicitly below.
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            probe.fit(train_x, train_y)
            accuracy = classification_accuracy(val_y, probe.predict(val_x))
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_c = c_value

    final_probe = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=best_c, max_iter=2000, random_state=seed),
    )
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        final_probe.fit(np.concatenate([train_x, val_x]), np.concatenate([train_y, val_y]))
        predictions = final_probe.predict(test_x)
    if not np.isfinite(final_probe[-1].coef_).all():
        raise FloatingPointError("Linear probe produced non-finite coefficients.")
    return {
        "accuracy": classification_accuracy(test_y, predictions),
        "macro_f1": macro_f1(test_y, predictions),
        "best_c": float(best_c),
        "num_train": int(len(train_y)),
        "num_val": int(len(val_y)),
        "num_test": int(len(test_y)),
        "num_classes": int(len(np.unique(np.concatenate([train_y, val_y, test_y])))),
    }


def content_heldout_indices(labels: np.ndarray, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split samples by class so speaker probes generalize across content."""
    classes = np.unique(labels)
    rng = np.random.default_rng(seed)
    classes = rng.permutation(classes)
    num_test = max(1, len(classes) // 5)
    num_val = max(1, len(classes) // 5)
    test_classes = set(classes[:num_test])
    val_classes = set(classes[num_test : num_test + num_val])
    train_classes = set(classes[num_test + num_val :])
    return (
        np.flatnonzero(np.isin(labels, list(train_classes))),
        np.flatnonzero(np.isin(labels, list(val_classes))),
        np.flatnonzero(np.isin(labels, list(test_classes))),
    )
