"""Discover RVTALL audio and attach exact Kinect repetitions to a manifest."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd


_INDEX_RE = re.compile(r"(\d+)$")


def _kinect_root(rvtall_base: str | Path) -> Path:
    base = Path(rvtall_base).expanduser().resolve()
    root = base if base.name == "kinect_processed" else base / "kinect_processed"
    if not root.is_dir():
        raise FileNotFoundError(f"RVTALL Kinect directory not found: {root}")
    return root


def _trailing_index(value: str) -> int | None:
    match = _INDEX_RE.search(Path(value).stem)
    return int(match.group(1)) if match else None


def _reference_repetitions(reference_npz: str | Path | None) -> dict[tuple[str, str], int]:
    if reference_npz is None:
        return {}
    path = Path(reference_npz).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Reference embedding file not found: {path}")
    with np.load(path, allow_pickle=True) as data:
        user_key = next((key for key in ("user_ids", "users", "speaker_ids") if key in data.files), None)
        group_key = next((key for key in ("group_names", "label_names", "sample_ids") if key in data.files), None)
        name_key = next((key for key in ("video_names", "sample_names") if key in data.files), None)
        if user_key is None or group_key is None or name_key is None:
            raise KeyError(
                f"{path} must contain user, group, and repetition-bearing sample names; found {data.files}"
            )
        users = data[user_key].astype(str)
        groups = data[group_key].astype(str)
        names = data[name_key].astype(str)

    repetitions: dict[tuple[str, str], int] = {}
    for user, group, name in zip(users, groups, names, strict=True):
        index = _trailing_index(name)
        if index is not None:
            # Embedding consumers currently index duplicate pairs by their last row.
            repetitions[(str(user), str(group))] = index
    return repetitions


def discover_rvtall_audio(rvtall_base: str | Path) -> dict[tuple[str, str], dict[int, Path]]:
    """Return available WAV repetitions keyed by ``(user_id, group_name)``."""
    root = _kinect_root(rvtall_base)
    discovered: dict[tuple[str, str], dict[int, Path]] = {}
    for path in root.glob("*/*/audios/audio_proc_*.wav"):
        index = _trailing_index(path.name)
        if index is None:
            continue
        user_id = path.parents[2].name
        group_name = path.parents[1].name
        discovered.setdefault((user_id, group_name), {})[index] = path.resolve()
    return discovered


def attach_rvtall_audio_paths(
    manifest: pd.DataFrame,
    rvtall_base: str | Path,
    reference_npz: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach one synchronized audio file to each manifest speaker/group pair.

    When a reference embedding is supplied, its final repetition for each duplicate
    pair is selected to mirror the repository's current embedding indexing behavior.
    Pairs without a reference use their latest available audio repetition.
    """
    required = {"user_id", "group_name"}
    missing_columns = required - set(manifest.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing_columns)}")

    audio = discover_rvtall_audio(rvtall_base)
    reference = _reference_repetitions(reference_npz)
    output = manifest.copy()
    paths: list[str] = []
    exact_reference_matches = 0
    fallback_latest_matches = 0
    missing_pairs: list[str] = []

    for row in output.itertuples(index=False):
        key = (str(row.user_id), str(row.group_name))
        candidates = audio.get(key, {})
        selected = None
        reference_index = reference.get(key)
        if reference_index is not None and reference_index in candidates:
            selected = candidates[reference_index]
            exact_reference_matches += 1
        elif candidates:
            selected = candidates[max(candidates)]
            fallback_latest_matches += 1
        else:
            missing_pairs.append(f"{key[0]}::{key[1]}")
        paths.append(str(selected) if selected else "")

    output["audio_path"] = paths
    covered = sum(bool(path) for path in paths)
    audit: dict[str, object] = {
        "manifest_rows": int(len(output)),
        "audio_files_discovered": int(sum(len(items) for items in audio.values())),
        "unique_audio_pairs_discovered": int(len(audio)),
        "rows_with_audio_path": int(covered),
        "coverage_fraction": float(covered / len(output)) if len(output) else 0.0,
        "exact_reference_matches": int(exact_reference_matches),
        "fallback_latest_matches": int(fallback_latest_matches),
        "missing_pair_count": int(len(missing_pairs)),
        "missing_pairs": missing_pairs,
        "reference_npz": str(Path(reference_npz).expanduser().resolve()) if reference_npz else "",
    }
    return output, audit
