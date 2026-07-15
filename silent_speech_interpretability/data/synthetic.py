"""Synthetic fixtures for tests and demos when real data is unavailable."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


MODALITIES = ("lip", "mouth", "uwb", "mmwave", "laser")


def make_synthetic_manifest(
    num_speakers: int = 20,
    classes_per_speaker: int = 30,
    repeats: int = 1,
) -> pd.DataFrame:
    rows = []
    for user_id in range(1, num_speakers + 1):
        for class_id in range(classes_per_speaker):
            for repeat in range(repeats):
                class_name = f"class_{class_id:02d}"
                sample_id = f"u{user_id:02d}_{class_name}_r{repeat:02d}"
                utterance_type = "vowel" if class_id < 5 else "word" if class_id < 20 else "sentence"
                row = {
                    "sample_id": sample_id,
                    "user_id": user_id,
                    "class_id": class_id,
                    "class_name": class_name,
                    "utterance_type": utterance_type,
                    "group_name": sample_id,
                    "audio_path": "",
                }
                for modality in MODALITIES:
                    row[f"{modality}_path" if modality != "mouth" else "mouth_video_path"] = ""
                rows.append(row)
    return pd.DataFrame(rows)


def make_synthetic_embeddings(
    output_dir: str | Path,
    manifest: pd.DataFrame,
    modalities: tuple[str, ...] = MODALITIES,
    embedding_dim: int = 128,
    seed: int = 42,
) -> dict[str, Path]:
    """Create class-structured synthetic embeddings keyed like the existing NPZ files."""
    rng = np.random.default_rng(seed)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    labels = manifest["class_id"].to_numpy(dtype=np.int64)
    user_ids = manifest["user_id"].to_numpy(dtype=np.int64)
    group_names = manifest["group_name"].astype(str).to_numpy()
    paths: dict[str, Path] = {}

    for modality_index, modality in enumerate(modalities):
        prototypes = rng.normal(size=(int(labels.max()) + 1, embedding_dim)).astype(np.float32)
        prototypes /= np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8
        noise = rng.normal(scale=0.35 + modality_index * 0.03, size=(len(labels), embedding_dim)).astype(np.float32)
        embeddings = prototypes[labels] + noise
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
        path = output / f"{modality}_embeddings_synthetic.npz"
        np.savez(path, embeddings=embeddings, labels=labels, user_ids=user_ids, group_names=group_names)
        paths[modality] = path
    return paths
