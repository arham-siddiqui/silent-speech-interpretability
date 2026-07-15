"""Manifest construction for existing embeddings and synthetic fallback data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .embeddings import GROUP_KEYS, USER_KEYS
from .synthetic import MODALITIES, make_synthetic_manifest
from .synthetic import make_synthetic_embeddings


MANIFEST_COLUMNS = [
    "sample_id",
    "user_id",
    "class_id",
    "class_name",
    "utterance_type",
    "lip_path",
    "mouth_video_path",
    "uwb_path",
    "mmwave_path",
    "laser_path",
    "audio_path",
    "group_name",
]


DEFAULT_EMBEDDING_GLOBS = {
    "lip": ["*lip*embeddings*.npz"],
    "mouth": ["*mouth*embeddings*.npz", "*mouth_frame_embeddings*.npz"],
    "uwb": ["*uwb*embeddings*.npz"],
    "mmwave": ["*radar*embeddings*.npz", "*mmwave*embeddings*.npz"],
    "laser": ["*laser*embeddings*.npz"],
}


def infer_utterance_type(class_id: int, class_name: str = "") -> str:
    lowered = class_name.lower()
    if "vowel" in lowered or class_id < 5:
        return "vowel"
    if "sentence" in lowered or class_id >= 20:
        return "sentence"
    return "word"


def discover_embedding_paths(search_roots: list[str | Path] | None = None) -> dict[str, Path]:
    roots = [Path(p) for p in (search_roots or [".", "artifacts/embeddings", "extra", "notebooks"])]
    found: dict[str, Path] = {}
    for modality, patterns in DEFAULT_EMBEDDING_GLOBS.items():
        for root in roots:
            if not root.exists():
                continue
            for pattern in patterns:
                matches = sorted(root.rglob(pattern))
                if matches:
                    found[modality] = matches[0]
                    break
            if modality in found:
                break
    return found


def configured_embedding_paths(config_data: Mapping[str, object]) -> dict[str, Path]:
    """Return explicit embedding paths from config, ignoring empty values."""
    configured = config_data.get("embedding_paths", {})
    if not isinstance(configured, Mapping):
        return {}
    paths = {}
    for modality in MODALITIES:
        value = configured.get(modality)
        if value:
            paths[modality] = Path(str(value)).expanduser()
    return paths


def resolve_embedding_paths(
    config_data: Mapping[str, object],
    search_roots: list[str | Path] | None = None,
    synthetic_if_missing: bool = True,
) -> tuple[dict[str, Path], dict[str, str]]:
    """Resolve embedding paths with precedence: config, discovery, synthetic fallback."""
    paths = configured_embedding_paths(config_data)
    sources = {modality: "configured" for modality in paths}

    discovery_roots = search_roots or [config_data.get("embeddings_dir", "artifacts/embeddings"), ".", "extra", "notebooks"]
    discovered = discover_embedding_paths(discovery_roots)
    for modality, path in discovered.items():
        if modality not in paths:
            paths[modality] = path
            sources[modality] = "discovered"

    if not paths and synthetic_if_missing:
        manifest = make_synthetic_manifest()
        synthetic_paths = make_synthetic_embeddings(config_data.get("embeddings_dir", "artifacts/embeddings"), manifest)
        paths = synthetic_paths
        sources = {modality: "synthetic" for modality in synthetic_paths}

    return paths, sources


def manifest_from_embeddings(embedding_paths: Mapping[str, str | Path]) -> pd.DataFrame:
    rows: dict[str, dict[str, object]] = {}
    for modality, raw_path in embedding_paths.items():
        path = Path(raw_path)
        with np.load(path, allow_pickle=True) as data:
            labels = data["labels"]
            user_key = next((key for key in USER_KEYS if key in data.files), None)
            group_key = next((key for key in GROUP_KEYS if key in data.files), None)
            if user_key is None or group_key is None:
                raise KeyError(f"Could not find user/group keys in {path}")
            user_ids = data[user_key].astype(str)
            group_names = data[group_key].astype(str)
        for idx, group_name in enumerate(group_names):
            user_id = str(user_ids[idx])
            group_name = str(group_name)
            sample_id = f"{user_id}::{group_name}" if group_name else f"{modality}_{idx:06d}"
            label = int(labels[idx])
            row = rows.setdefault(
                sample_id,
                {
                    "sample_id": sample_id,
                    "user_id": int(user_id),
                    "class_id": label,
                    "class_name": f"class_{label:02d}",
                    "utterance_type": infer_utterance_type(label),
                    "lip_path": "",
                    "mouth_video_path": "",
                    "uwb_path": "",
                    "mmwave_path": "",
                    "laser_path": "",
                    "audio_path": "",
                    "group_name": group_name,
                },
            )
            if int(row["class_id"]) != label:
                raise ValueError(f"Inconsistent labels for {sample_id} while reading {path}")
            key = "mouth_video_path" if modality == "mouth" else f"{modality}_path"
            row[key] = str(path)
    if not rows:
        return pd.DataFrame(columns=MANIFEST_COLUMNS)
    return pd.DataFrame(rows.values())[MANIFEST_COLUMNS].sort_values(["user_id", "class_id", "sample_id"])


def build_intersection_manifest(manifest: pd.DataFrame, modalities: tuple[str, ...] = MODALITIES) -> pd.DataFrame:
    mask = pd.Series(True, index=manifest.index)
    for modality in modalities:
        column = "mouth_video_path" if modality == "mouth" else f"{modality}_path"
        if column in manifest:
            mask &= manifest[column].fillna("").astype(str).ne("")
    return manifest.loc[mask].copy()


def _load_embedding_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=True) as data:
        labels = data["labels"].astype(np.int64)
        user_key = next((key for key in USER_KEYS if key in data.files), None)
        group_key = next((key for key in GROUP_KEYS if key in data.files), None)
        if user_key is None or group_key is None:
            raise KeyError(f"Could not find user/group keys in {path}")
        user_ids = data[user_key].astype(str)
        group_names = data[group_key].astype(str)
        embedding_shape = list(data["embeddings"].shape)
    pair_ids = np.asarray([f"{user}::{group}" for user, group in zip(user_ids, group_names)])
    pair_label = {}
    pair_user = {}
    for pair_id, label, user_id in zip(pair_ids, labels, user_ids):
        pair_label.setdefault(pair_id, int(label))
        pair_user.setdefault(pair_id, int(user_id))
        if pair_label[pair_id] != int(label):
            raise ValueError(f"Inconsistent labels inside pair {pair_id} in {path}")
    return {
        "labels": labels,
        "user_ids": user_ids.astype(str),
        "group_names": group_names,
        "pair_ids": pair_ids,
        "pair_label": pair_label,
        "pair_user": pair_user,
        "embedding_shape": embedding_shape,
        "index": {pair_id: i for i, pair_id in enumerate(pair_ids)},
    }


def build_alignment_audit(
    embedding_paths: Mapping[str, str | Path],
    path_sources: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Summarize multimodal alignment by group name, label, and speaker ID."""
    metadata = {}
    for modality, raw_path in embedding_paths.items():
        path = Path(raw_path)
        if path.exists():
            metadata[modality] = _load_embedding_metadata(path)

    all_groups = sorted(set().union(*(set(item["pair_ids"]) for item in metadata.values()))) if metadata else []
    intersection_groups = sorted(set.intersection(*(set(item["pair_ids"]) for item in metadata.values()))) if metadata else []

    modalities = sorted(set(MODALITIES) | set(embedding_paths))
    by_modality = {}
    for modality in modalities:
        raw_path = embedding_paths.get(modality)
        path = Path(raw_path) if raw_path else None
        item = metadata.get(modality)
        if item is None:
            by_modality[modality] = {
                "path": str(path) if path else "",
                "source": (path_sources or {}).get(modality, "missing"),
                "exists": bool(path and path.exists()),
                "num_rows": 0,
                "unique_groups": 0,
                "duplicate_groups": 0,
                "num_speakers": 0,
                "num_classes": 0,
                "missing_from_union": len(all_groups),
            }
            continue
        groups = item["group_names"]
        pairs = item["pair_ids"]
        by_modality[modality] = {
            "path": str(path),
            "source": (path_sources or {}).get(modality, "unknown"),
            "exists": True,
            "embedding_shape": item["embedding_shape"],
            "num_rows": int(len(groups)),
            "unique_groups": int(len(set(pairs))),
            "duplicate_groups": int(len(pairs) - len(set(pairs))),
            "num_speakers": int(len(set(item["user_ids"]))),
            "num_classes": int(len(set(item["labels"]))),
            "missing_from_union": int(len(set(all_groups) - set(pairs))),
        }

    label_mismatches = []
    user_mismatches = []
    modalities_present = sorted(metadata)
    if modalities_present and intersection_groups:
        reference_modality = modalities_present[0]
        reference = metadata[reference_modality]
        for modality in modalities_present[1:]:
            item = metadata[modality]
            for group in intersection_groups:
                if int(reference["pair_label"][group]) != int(item["pair_label"][group]):
                    label_mismatches.append(
                        {
                            "group_name": group,
                            "reference_modality": reference_modality,
                            "modality": modality,
                            "reference_label": int(reference["pair_label"][group]),
                            "label": int(item["pair_label"][group]),
                        }
                    )
                if int(reference["pair_user"][group]) != int(item["pair_user"][group]):
                    user_mismatches.append(
                        {
                            "group_name": group,
                            "reference_modality": reference_modality,
                            "modality": modality,
                            "reference_user_id": int(reference["pair_user"][group]),
                            "user_id": int(item["pair_user"][group]),
                        }
                    )

    return {
        "embedding_paths": by_modality,
        "union_group_count": int(len(all_groups)),
        "strict_intersection_group_count": int(len(intersection_groups)),
        "label_mismatch_count": int(len(label_mismatches)),
        "user_id_mismatch_count": int(len(user_mismatches)),
        "label_mismatch_examples": label_mismatches[:20],
        "user_id_mismatch_examples": user_mismatches[:20],
    }


def write_dataset_audit(
    manifest: pd.DataFrame,
    output_path: str | Path,
    embedding_paths: Mapping[str, str | Path] | None = None,
    path_sources: Mapping[str, str] | None = None,
) -> dict[str, object]:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    modality_counts = {}
    modality_missing = {}
    for modality in MODALITIES:
        column = "mouth_video_path" if modality == "mouth" else f"{modality}_path"
        modality_counts[modality] = int(manifest[column].fillna("").astype(str).ne("").sum()) if column in manifest else 0
        modality_missing[modality] = int(manifest[column].fillna("").astype(str).eq("").sum()) if column in manifest else int(len(manifest))
    intersection = build_intersection_manifest(manifest)
    audit = {
        "num_samples": int(len(manifest)),
        "strict_intersection_samples": int(len(intersection)),
        "num_speakers": int(manifest["user_id"].nunique()) if len(manifest) else 0,
        "num_classes": int(manifest["class_id"].nunique()) if len(manifest) else 0,
        "modality_counts": modality_counts,
        "missing_by_modality": modality_missing,
        "samples_by_speaker": {str(k): int(v) for k, v in manifest["user_id"].value_counts().sort_index().items()},
        "samples_by_class": {str(k): int(v) for k, v in manifest["class_id"].value_counts().sort_index().items()},
    }
    if embedding_paths is not None:
        audit["alignment"] = build_alignment_audit(embedding_paths, path_sources)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def enforce_quality_gates(audit: Mapping[str, object], gates: Mapping[str, object]) -> None:
    """Raise if dataset/embedding alignment fails configured quality gates."""
    alignment = audit.get("alignment", {})
    embedding_paths = alignment.get("embedding_paths", {}) if isinstance(alignment, Mapping) else {}
    loaded = [
        modality
        for modality, info in embedding_paths.items()
        if isinstance(info, Mapping) and info.get("exists") and int(info.get("num_rows", 0)) > 0
    ]
    synthetic = [
        modality
        for modality, info in embedding_paths.items()
        if isinstance(info, Mapping) and info.get("source") == "synthetic"
    ]
    failures = []

    min_modalities = int(gates.get("min_modalities", 1))
    if len(loaded) < min_modalities:
        failures.append(f"loaded {len(loaded)} modalities, expected at least {min_modalities}")

    min_intersection = int(gates.get("min_strict_intersection", 1))
    strict_intersection = int(alignment.get("strict_intersection_group_count", audit.get("strict_intersection_samples", 0)))
    if strict_intersection < min_intersection:
        failures.append(f"strict intersection has {strict_intersection} samples, expected at least {min_intersection}")

    if not bool(gates.get("allow_synthetic", True)) and synthetic:
        failures.append(f"synthetic embeddings are not allowed but were used for: {', '.join(sorted(synthetic))}")

    if bool(gates.get("fail_on_label_mismatch", True)) and int(alignment.get("label_mismatch_count", 0)) > 0:
        failures.append(f"{alignment.get('label_mismatch_count')} label mismatches found")

    if bool(gates.get("fail_on_user_id_mismatch", True)) and int(alignment.get("user_id_mismatch_count", 0)) > 0:
        failures.append(f"{alignment.get('user_id_mismatch_count')} user ID mismatches found")

    if failures:
        raise RuntimeError("Dataset quality gates failed: " + "; ".join(failures))


def build_manifest(
    embedding_paths: Mapping[str, str | Path] | None = None,
    synthetic_if_missing: bool = True,
) -> pd.DataFrame:
    paths = dict(embedding_paths or discover_embedding_paths())
    manifest = manifest_from_embeddings(paths)
    if manifest.empty and synthetic_if_missing:
        manifest = make_synthetic_manifest()
    return manifest
