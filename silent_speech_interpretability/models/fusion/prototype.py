"""Cosine prototype classifier for speaker-disjoint baselines."""

from __future__ import annotations

import numpy as np


def _l2_normalize(x: np.ndarray, axis: int = 1) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + 1e-8)


def _softmax(x: np.ndarray, axis: int = 1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-8)


class PrototypeClassifier:
    def __init__(self, temperature: float = 0.07):
        self.temperature = temperature
        self.classes_: np.ndarray | None = None
        self.prototypes_: np.ndarray | None = None

    def fit(self, embeddings: np.ndarray, labels: np.ndarray) -> "PrototypeClassifier":
        embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        labels = np.asarray(labels)
        classes = np.array(sorted(np.unique(labels)))
        prototypes = []
        for cls in classes:
            prototype = embeddings[labels == cls].mean(axis=0)
            prototypes.append(prototype)
        self.classes_ = classes
        self.prototypes_ = _l2_normalize(np.stack(prototypes, axis=0))
        return self

    def predict_proba(self, embeddings: np.ndarray) -> np.ndarray:
        if self.prototypes_ is None:
            raise RuntimeError("PrototypeClassifier.fit must be called before predict_proba.")
        embeddings = _l2_normalize(np.asarray(embeddings, dtype=np.float32))
        logits = np.einsum("nd,cd->nc", embeddings, self.prototypes_) / self.temperature
        return _softmax(logits)

    def predict(self, embeddings: np.ndarray) -> np.ndarray:
        if self.classes_ is None:
            raise RuntimeError("PrototypeClassifier.fit must be called before predict.")
        proba = self.predict_proba(embeddings)
        return self.classes_[np.argmax(proba, axis=1)]
