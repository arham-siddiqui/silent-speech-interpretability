"""Speaker-disjoint split construction."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def validate_speaker_disjoint(*dfs: pd.DataFrame) -> None:
    speaker_sets = [set(df["user_id"].astype(int).unique()) for df in dfs]
    for i, first in enumerate(speaker_sets):
        for j, second in enumerate(speaker_sets):
            if i >= j:
                continue
            overlap = first & second
            if overlap:
                raise ValueError(f"Speaker overlap between split {i} and {j}: {sorted(overlap)}")


def make_fixed_split(
    manifest: pd.DataFrame,
    train_speakers: list[int],
    val_speakers: list[int],
    test_speakers: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = manifest[manifest["user_id"].isin(train_speakers)].copy()
    val = manifest[manifest["user_id"].isin(val_speakers)].copy()
    test = manifest[manifest["user_id"].isin(test_speakers)].copy()
    validate_speaker_disjoint(train, val, test)
    return train, val, test


def summarize_class_distribution(df: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in df["class_id"].value_counts().sort_index().items()}


def make_speaker_kfold_splits(
    manifest: pd.DataFrame,
    num_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, object]]:
    speakers = np.array(sorted(manifest["user_id"].astype(int).unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    speaker_folds = [list(map(int, fold)) for fold in np.array_split(speakers, num_folds)]
    splits = []
    for fold_index, test_speakers in enumerate(speaker_folds):
        remaining = [int(s) for s in speakers if int(s) not in set(test_speakers)]
        val_count = max(1, len(test_speakers) // 2)
        val_speakers = remaining[:val_count]
        train_speakers = [s for s in remaining if s not in set(val_speakers)]
        train, val, test = make_fixed_split(manifest, train_speakers, val_speakers, test_speakers)
        splits.append(
            {
                "fold": fold_index,
                "train_speakers": train_speakers,
                "val_speakers": val_speakers,
                "test_speakers": test_speakers,
                "num_train": int(len(train)),
                "num_val": int(len(val)),
                "num_test": int(len(test)),
                "class_distribution": {
                    "train": summarize_class_distribution(train),
                    "val": summarize_class_distribution(val),
                    "test": summarize_class_distribution(test),
                },
            }
        )
    return splits


def save_split_json(payload: object, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
