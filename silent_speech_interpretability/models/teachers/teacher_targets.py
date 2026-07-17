"""Teacher target indexing and storage helpers.

Teacher targets are fixed-dimensional vectors keyed by `(user_id, group_name)`.
They can come from real audio teachers later, or from deterministic synthetic
targets for local smoke tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np


def save_teacher_targets(
    path: str | Path,
    targets: np.ndarray,
    labels: np.ndarray,
    user_ids: np.ndarray,
    group_names: np.ndarray,
    *,
    target_name: str = "synthetic_teacher",
    class_names: np.ndarray | None = None,
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "targets": np.asarray(targets, dtype=np.float32),
        "labels": np.asarray(labels, dtype=np.int64),
        "user_ids": np.asarray(user_ids).astype(str),
        "group_names": np.asarray(group_names).astype(str),
        "target_name": np.asarray(target_name),
    }
    if class_names is not None:
        payload["class_names"] = np.asarray(class_names).astype(str)
    np.savez_compressed(output, **payload)
    return output


def load_teacher_targets(path: str | Path) -> dict[str, object]:
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        targets = data["targets"].astype(np.float32)
        labels = data["labels"].astype(np.int64)
        user_ids = data["user_ids"].astype(str)
        group_names = data["group_names"].astype(str)
        target_name = str(data["target_name"]) if "target_name" in data.files else path.stem

    if not (len(targets) == len(labels) == len(user_ids) == len(group_names)):
        raise ValueError(f"Inconsistent teacher target lengths in {path}")

    index = {}
    label_by_pair = {}
    for i, pair in enumerate(zip(user_ids, group_names, strict=True)):
        if pair in index:
            raise ValueError(f"Duplicate teacher target pair {pair} in {path}")
        index[pair] = i
        label_by_pair[pair] = int(labels[i])
    return {
        "path": path,
        "target_name": target_name,
        "targets": targets,
        "labels": labels,
        "user_ids": user_ids,
        "group_names": group_names,
        "pairs": set(index),
        "index": index,
        "label_by_pair": label_by_pair,
        "target_dim": int(targets.shape[1]),
    }


def common_teacher_pairs(
    embedding_payloads: Mapping[str, dict[str, object]],
    teacher_payload: dict[str, object],
    speakers: list[int] | list[str] | None = None,
) -> list[tuple[str, str]]:
    speaker_set = {str(speaker) for speaker in speakers} if speakers is not None else None
    pair_sets = [set(payload["pairs"]) for payload in embedding_payloads.values()]
    pair_sets.append(set(teacher_payload["pairs"]))
    pairs = set.intersection(*pair_sets) if pair_sets else set()
    if speaker_set is not None:
        pairs = {pair for pair in pairs if pair[0] in speaker_set}
    return sorted(pairs)


def teacher_arrays(teacher_payload: dict[str, object], pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    indices = [teacher_payload["index"][pair] for pair in pairs]
    return teacher_payload["targets"][indices].astype(np.float32), teacher_payload["labels"][indices].astype(np.int64)


def make_class_structured_targets(
    labels: np.ndarray,
    target_dim: int = 64,
    seed: int = 42,
    noise: float = 0.02,
) -> np.ndarray:
    """Make deterministic class-structured targets for tests and smoke runs."""
    labels = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    prototypes = rng.normal(size=(int(labels.max()) + 1, target_dim)).astype(np.float32)
    prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
    targets = prototypes[labels] + rng.normal(scale=noise, size=(len(labels), target_dim)).astype(np.float32)
    return targets.astype(np.float32)
