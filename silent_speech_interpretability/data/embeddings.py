"""Compatibility loaders for legacy pre-computed embedding NPZ files."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np


USER_KEYS = ("user_ids", "users")
GROUP_KEYS = ("group_names", "label_names")
SORT_KEYS = ("sample_names", "video_names")


def natural_sort_key(value: str) -> list[object]:
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", value)]


def _first_present(data: np.lib.npyio.NpzFile, keys: tuple[str, ...], path: Path) -> str:
    for key in keys:
        if key in data.files:
            return key
    raise KeyError(f"None of {keys} found in {path}")


def load_embedding_repetitions(path: str | Path) -> dict[str, object]:
    """Load an NPZ as grouped repetitions keyed by `(user_id, group_name)`.

    The legacy files are not perfectly schema-consistent: mouth embeddings use
    `users` and `label_names`; other modalities usually use `user_ids` and
    `group_names`. Several files also contain many repetitions per group, so
    raw `group_names` are not unique sample IDs.
    """
    path = Path(path)
    with np.load(path, allow_pickle=True) as data:
        user_key = _first_present(data, USER_KEYS, path)
        group_key = _first_present(data, GROUP_KEYS, path)
        user_ids = data[user_key].astype(str)
        group_names = data[group_key].astype(str)
        labels = data["labels"].astype(np.int64)
        embeddings = data["embeddings"].astype(np.float32)
        sort_values = np.arange(len(embeddings)).astype(str)
        for sort_key in SORT_KEYS:
            if sort_key in data.files:
                sort_values = data[sort_key].astype(str)
                break

    grouped = defaultdict(list)
    label_by_pair = {}
    for idx, embedding in enumerate(embeddings):
        pair = (str(user_ids[idx]), str(group_names[idx]))
        grouped[pair].append((str(sort_values[idx]), embedding))
        label_by_pair.setdefault(pair, int(labels[idx]))
        if label_by_pair[pair] != int(labels[idx]):
            raise ValueError(f"Inconsistent labels inside pair {pair} in {path}")

    repetitions = {
        pair: [embedding for _sort_value, embedding in sorted(items, key=lambda item: natural_sort_key(item[0]))]
        for pair, items in grouped.items()
    }
    return {
        "path": path,
        "repetitions": repetitions,
        "labels": label_by_pair,
        "pairs": set(repetitions),
        "num_rows": int(len(embeddings)),
    }


def common_pairs(payloads: Mapping[str, dict[str, object]], speakers: list[int] | list[str] | None = None) -> list[tuple[str, str]]:
    speaker_set = {str(speaker) for speaker in speakers} if speakers is not None else None
    pair_sets = []
    for payload in payloads.values():
        pairs = set(payload["pairs"])
        if speaker_set is not None:
            pairs = {pair for pair in pairs if pair[0] in speaker_set}
        pair_sets.append(pairs)
    return sorted(set.intersection(*pair_sets)) if pair_sets else []


def modality_pairs(payload: dict[str, object], speakers: list[int] | list[str]) -> list[tuple[str, str]]:
    speaker_set = {str(speaker) for speaker in speakers}
    return sorted(pair for pair in payload["pairs"] if pair[0] in speaker_set)


def repetition_training_arrays(payload: dict[str, object], pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    embeddings = []
    labels = []
    for pair in pairs:
        reps = payload["repetitions"][pair]
        embeddings.extend(reps)
        labels.extend([payload["labels"][pair]] * len(reps))
    return np.stack(embeddings).astype(np.float32), np.asarray(labels, dtype=np.int64)


def mean_eval_arrays(payload: dict[str, object], pairs: list[tuple[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    embeddings = []
    labels = []
    for pair in pairs:
        embeddings.append(np.mean(payload["repetitions"][pair], axis=0))
        labels.append(payload["labels"][pair])
    return np.stack(embeddings).astype(np.float32), np.asarray(labels, dtype=np.int64)


def validate_pair_labels(payloads: Mapping[str, dict[str, object]], pairs: list[tuple[str, str]]) -> np.ndarray:
    modalities = list(payloads)
    reference = payloads[modalities[0]]
    labels = np.asarray([reference["labels"][pair] for pair in pairs], dtype=np.int64)
    for modality in modalities[1:]:
        other = np.asarray([payloads[modality]["labels"][pair] for pair in pairs], dtype=np.int64)
        if not np.array_equal(labels, other):
            raise ValueError(f"Label mismatch across modalities involving {modality!r}.")
    return labels
