#!/usr/bin/env python3
"""Extract utterance-level SSL audio teacher targets from manifest audio paths."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.models.teachers.ssl_teacher import SSLTeacher
from silent_speech_interpretability.models.teachers.teacher_targets import save_teacher_targets


def _load_manifest(config: dict, manifest_path: str | Path | None) -> pd.DataFrame:
    if manifest_path:
        return pd.read_csv(manifest_path)
    configured = Path(config["data"].get("manifest_path", "artifacts/manifest.csv"))
    if configured.exists():
        return pd.read_csv(configured)
    seed_paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    return build_manifest(seed_paths)


def _audio_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    if "audio_path" not in manifest.columns:
        raise ValueError("Manifest has no audio_path column.")
    rows = manifest.copy()
    rows["audio_path"] = rows["audio_path"].fillna("").astype(str)
    rows = rows[rows["audio_path"].ne("")]
    rows = rows[rows["audio_path"].map(lambda value: Path(value).expanduser().exists())]
    if rows.empty:
        return rows
    return rows.drop_duplicates(["user_id", "group_name"]).sort_values(["user_id", "group_name"])


def _write_audit(manifest: pd.DataFrame, rows: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audit = {
        "manifest_rows": int(len(manifest)),
        "rows_with_audio_path": int(manifest.get("audio_path", pd.Series([], dtype=str)).fillna("").astype(str).ne("").sum())
        if "audio_path" in manifest
        else 0,
        "existing_audio_files": int(len(rows)),
        "unique_audio_pairs": int(len(rows.drop_duplicates(["user_id", "group_name"]))) if not rows.empty else 0,
    }
    pd.DataFrame([audit]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--model-name", default="facebook/hubert-base-ls960")
    parser.add_argument("--output", default=None)
    parser.add_argument("--audit-output", default="reports/results/ssl_teacher_audio_audit.csv")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    manifest = _load_manifest(config, args.manifest)
    rows = _audio_rows(manifest)
    if args.limit:
        rows = rows.head(args.limit)
    _write_audit(manifest, rows, Path(args.audit_output))
    print(f"Audio audit: {len(rows)} usable audio rows; wrote {args.audit_output}")

    if args.dry_run:
        return
    if rows.empty:
        raise RuntimeError("No usable audio files found. Populate manifest audio_path before SSL extraction.")

    teacher = SSLTeacher(args.model_name, device=args.device, local_files_only=args.local_files_only)
    if not teacher.available():
        raise RuntimeError("Missing optional dependencies for SSL extraction. Install transformers and librosa.")
    pooled = []
    labels = []
    users = []
    groups = []
    for row in rows.itertuples(index=False):
        extracted = teacher.extract_hidden_states(row.audio_path)
        pooled.append(extracted["pooled"])
        labels.append(int(row.class_id))
        users.append(str(row.user_id))
        groups.append(str(row.group_name))

    output = Path(args.output or Path(config["data"]["teacher_targets_dir"]) / f"{args.model_name.replace('/', '_')}_targets.npz")
    save_teacher_targets(
        output,
        np.stack(pooled).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(users),
        np.asarray(groups),
        target_name=args.model_name,
    )
    print(f"Saved {len(pooled)} SSL teacher targets to {output}")


if __name__ == "__main__":
    main()
