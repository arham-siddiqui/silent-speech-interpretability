"""Helpers for true encoder-disjoint cross-validation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from silent_speech_interpretability.data.synthetic import MODALITIES


def expected_fold_embedding_paths(
    embeddings_dir: str | Path,
    fold: int,
    modalities: list[str] | tuple[str, ...] = MODALITIES,
) -> dict[str, Path]:
    root = Path(embeddings_dir)
    return {modality: root / f"fold_{fold}" / f"{modality}_embeddings.npz" for modality in modalities}


def configured_fold_embedding_paths(
    true_cv_config: Mapping[str, object],
    fold: int,
    modalities: list[str] | tuple[str, ...] = MODALITIES,
) -> dict[str, Path]:
    configured = true_cv_config.get("fold_embedding_paths", {})
    fold_key = f"fold_{fold}"
    if isinstance(configured, Mapping) and isinstance(configured.get(fold_key), Mapping):
        paths = {
            modality: Path(str(configured[fold_key][modality])).expanduser()
            for modality in modalities
            if configured[fold_key].get(modality)
        }
        if paths:
            return paths
    return expected_fold_embedding_paths(true_cv_config.get("embeddings_dir", "artifacts/embeddings/speaker_cv"), fold, modalities)


def missing_embedding_paths(paths: Mapping[str, str | Path]) -> dict[str, str]:
    return {modality: str(path) for modality, path in paths.items() if not Path(path).exists()}


def metadata_path_for_fold(embeddings_dir: str | Path, fold: int) -> Path:
    return Path(embeddings_dir) / f"fold_{fold}" / "metadata.json"
