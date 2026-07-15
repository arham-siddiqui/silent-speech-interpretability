"""Shared metadata helpers for fold-specific encoder artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from silent_speech_interpretability.evals.true_cv import metadata_path_for_fold

EXPECTED_MODALITIES = ["lip", "mouth", "uwb", "mmwave", "laser"]


def load_or_create_fold_metadata(config: dict[str, Any], fold: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    embeddings_dir = Path(config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv"))
    path = metadata_path_for_fold(embeddings_dir, int(fold["fold"]))
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "fold": int(fold["fold"]),
            "train_speakers": fold["train_speakers"],
            "val_speakers": fold["val_speakers"],
            "test_speakers": fold["test_speakers"],
            "modalities": EXPECTED_MODALITIES,
            "status": "planned",
        }
    metadata.setdefault("completed_modalities", [])
    metadata.setdefault("modalities", EXPECTED_MODALITIES)
    return path, metadata


def mark_modality_complete(
    metadata_path: Path,
    metadata: dict[str, Any],
    modality: str,
    embedding_path: Path,
    checkpoint_path: Path,
    label_map_path: Path,
    training_metadata: dict[str, Any],
) -> None:
    completed = set(metadata.get("completed_modalities", []))
    completed.add(modality)
    metadata["completed_modalities"] = sorted(completed)
    metadata[f"{modality}_embedding_path"] = str(embedding_path)
    metadata[f"{modality}_checkpoint_path"] = str(checkpoint_path)
    metadata[f"{modality}_label_map_path"] = str(label_map_path)
    metadata[f"{modality}_training"] = {
        **training_metadata,
        "note": "Short max_epochs values are smoke tests, not final scientific fold embeddings.",
    }
    expected = set(metadata.get("modalities", EXPECTED_MODALITIES))
    metadata["status"] = "completed" if expected and expected.issubset(completed) else "partial"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
