#!/usr/bin/env python3
"""Prepare the fold-specific artifact plan needed for true encoder-disjoint CV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits, save_split_json
from silent_speech_interpretability.data.synthetic import MODALITIES
from silent_speech_interpretability.evals.true_cv import expected_fold_embedding_paths, metadata_path_for_fold


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    save_split_json(folds, "artifacts/splits/true_encoder_speaker_kfold_5.json")

    true_cv = config.get("true_encoder_cv", {})
    embeddings_dir = Path(true_cv.get("embeddings_dir", "artifacts/embeddings/speaker_cv"))
    plan_rows = []
    report = [
        "# True Encoder-Disjoint CV Preparation Plan",
        "",
        "Each fold needs encoders trained only on that fold's train speakers, then embeddings extracted for all speakers using those fold-specific encoders.",
        "",
        "Expected artifacts per fold:",
        "",
    ]

    for fold in folds:
        fold_id = int(fold["fold"])
        expected = expected_fold_embedding_paths(embeddings_dir, fold_id)
        fold_dir = embeddings_dir / f"fold_{fold_id}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "fold": fold_id,
            "train_speakers": fold["train_speakers"],
            "val_speakers": fold["val_speakers"],
            "test_speakers": fold["test_speakers"],
            "modalities": list(MODALITIES),
            "status": "planned",
            "note": "Replace status with completed after fold-specific encoders are trained and embeddings are extracted.",
        }
        metadata_path = metadata_path_for_fold(embeddings_dir, fold_id)
        if not metadata_path.exists():
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

        report.extend(
            [
                f"## Fold {fold_id}",
                "",
                f"- Train speakers: {fold['train_speakers']}",
                f"- Val speakers: {fold['val_speakers']}",
                f"- Test speakers: {fold['test_speakers']}",
                f"- Metadata: `{metadata_path}`",
                "- Lip command:",
                (
                    "  `python3 scripts/09_train_lip_fold_embeddings.py "
                    f"--config configs/real_embeddings.local.yaml --fold {fold_id}`"
                ),
                "",
            ]
        )
        for modality, path in expected.items():
            plan_rows.append(
                {
                    "fold": fold_id,
                    "modality": modality,
                    "expected_embedding_path": str(path),
                    "exists": path.exists(),
                }
            )
            report.append(f"- `{path}`")
        report.append("")

    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    output = reports_dir / "true_encoder_cv_plan.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Prepared artifact directories under {embeddings_dir}")


if __name__ == "__main__":
    main()
