#!/usr/bin/env python3
"""Run fold-specific encoder artifact generation with resume-friendly logging."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.evals.true_cv import expected_fold_embedding_paths, metadata_path_for_fold

MODALITY_SCRIPTS = {
    "lip": "scripts/09_train_lip_fold_embeddings.py",
    "laser": "scripts/10_train_laser_fold_embeddings.py",
    "mouth": "scripts/11_train_mouth_fold_embeddings.py",
    "uwb": "scripts/12_train_uwb_fold_embeddings.py",
    "mmwave": "scripts/13_train_mmwave_fold_embeddings.py",
}

MIN_EPOCH_KEYS = {
    "lip": "min_lip_epochs",
    "mouth": "min_mouth_epochs",
    "uwb": "min_uwb_epochs",
    "mmwave": "min_mmwave_epochs",
    "laser": "min_laser_epochs",
}


def _metadata(config: dict, fold: int) -> dict:
    embeddings_dir = config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv")
    path = metadata_path_for_fold(embeddings_dir, fold)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _is_valid(config: dict, fold: int, modality: str) -> bool:
    true_cv = config.get("true_encoder_cv", {})
    path = expected_fold_embedding_paths(true_cv.get("embeddings_dir", "artifacts/embeddings/speaker_cv"), fold, (modality,))[modality]
    if not path.exists():
        return False
    metadata = _metadata(config, fold)
    training = metadata.get(f"{modality}_training", {})
    min_epochs = int(true_cv.get(MIN_EPOCH_KEYS[modality], 0))
    return int(training.get("max_epochs", 0)) >= min_epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--modalities", default="lip,laser,uwb,mmwave,mouth")
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-dir", default="reports/results/true_encoder_cv_logs")
    parser.add_argument("--force", action="store_true", help="Re-run artifacts even when they pass the epoch gate.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    folds = [int(item.strip()) for item in args.folds.split(",") if item.strip()]
    modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    unknown = sorted(set(modalities) - set(MODALITY_SCRIPTS))
    if unknown:
        raise ValueError(f"Unknown modalities: {unknown}")

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    for fold in folds:
        for modality in modalities:
            if not args.force and _is_valid(config, fold, modality):
                print(f"SKIP fold={fold} modality={modality}: artifact already passes epoch gate", flush=True)
                continue
            command = [sys.executable, MODALITY_SCRIPTS[modality], "--config", args.config, "--fold", str(fold)]
            if args.device:
                command.extend(["--device", args.device])
            if modality in {"lip", "laser"}:
                command.append("--resume")
            log_path = log_dir / f"fold_{fold}_{modality}.log"
            print(f"RUN fold={fold} modality={modality}: {' '.join(command)}", flush=True)
            print(f"LOG {log_path}", flush=True)
            if args.dry_run:
                continue
            with log_path.open("w", encoding="utf-8") as log_file:
                proc = subprocess.run(command, stdout=log_file, stderr=subprocess.STDOUT, text=True)
            if proc.returncode != 0:
                raise RuntimeError(f"Command failed with code {proc.returncode}; see {log_path}")


if __name__ == "__main__":
    main()
