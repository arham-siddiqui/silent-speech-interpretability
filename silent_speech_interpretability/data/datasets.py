"""PyTorch datasets for pre-computed utterance embeddings."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        embedding_paths: Mapping[str, str | Path],
        manifest_path: str | Path | None = None,
        modalities: list[str] | None = None,
        strict_intersection: bool = True,
        allow_missing: bool = False,
    ):
        if strict_intersection and allow_missing:
            raise ValueError("Choose either strict_intersection or allow_missing, not both.")
        self.modalities = modalities or list(embedding_paths.keys())
        self.strict_intersection = strict_intersection
        self.allow_missing = allow_missing
        self.by_modality = {m: self._load_npz(Path(embedding_paths[m])) for m in self.modalities if m in embedding_paths}

        group_sets = [set(payload["group_names"]) for payload in self.by_modality.values()]
        if strict_intersection:
            group_names = sorted(set.intersection(*group_sets)) if group_sets else []
        else:
            group_names = sorted(set.union(*group_sets)) if group_sets else []
        self.samples = group_names
        self.manifest_path = Path(manifest_path) if manifest_path else None

    @staticmethod
    def _load_npz(path: Path) -> dict[str, object]:
        with np.load(path, allow_pickle=True) as data:
            embeddings = data["embeddings"].astype(np.float32)
            labels = data["labels"].astype(np.int64)
            user_ids = data["user_ids"].astype(np.int64)
            group_names = data["group_names"].astype(str)
        index = {group_name: i for i, group_name in enumerate(group_names)}
        return {
            "embeddings": embeddings,
            "labels": labels,
            "user_ids": user_ids,
            "group_names": group_names,
            "index": index,
            "dim": embeddings.shape[1],
        }

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        group_name = self.samples[idx]
        embeddings = {}
        mask = {}
        label = None
        user_id = None
        for modality in self.modalities:
            payload = self.by_modality.get(modality)
            item_index = payload["index"].get(group_name) if payload else None
            if item_index is None:
                if self.strict_intersection or not self.allow_missing:
                    raise KeyError(f"Missing modality {modality!r} for sample {group_name!r}")
                dim = int(payload["dim"]) if payload else 128
                embeddings[modality] = torch.zeros(dim, dtype=torch.float32)
                mask[modality] = False
                continue
            embeddings[modality] = torch.from_numpy(payload["embeddings"][item_index])
            mask[modality] = True
            label = int(payload["labels"][item_index]) if label is None else label
            user_id = int(payload["user_ids"][item_index]) if user_id is None else user_id

        return {
            "sample_id": group_name,
            "user_id": int(user_id) if user_id is not None else -1,
            "label": int(label) if label is not None else -1,
            "group_name": group_name,
            "embeddings": embeddings,
            "modality_mask": mask,
        }
