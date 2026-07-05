#!/usr/bin/env python3
"""Run 5-fold speaker-disjoint cross-validation on embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, discover_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits, save_split_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    manifest = build_manifest(discover_embedding_paths([config["data"]["embeddings_dir"], ".", "extra", "notebooks"]))
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    save_split_json(folds, "artifacts/splits/speaker_kfold_5.json")

    rows = []
    for fold in folds:
        majority_class = manifest[manifest["user_id"].isin(fold["train_speakers"])]["class_id"].mode().iloc[0]
        test = manifest[manifest["user_id"].isin(fold["test_speakers"])]
        y_true = test["class_id"].to_numpy()
        y_pred = np.full_like(y_true, majority_class)
        rows.append(
            {
                "fold": fold["fold"],
                "method": "majority_sanity",
                "modality": "manifest",
                "accuracy": float(np.mean(y_true == y_pred)) if len(y_true) else 0.0,
                "macro_f1": 0.0,
                "num_train": fold["num_train"],
                "num_val": fold["num_val"],
                "num_test": fold["num_test"],
                "test_speakers": ",".join(map(str, fold["test_speakers"])),
            }
        )
    results = pd.DataFrame(rows)
    results_dir = Path(config["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "speaker_cv_results.csv", index=False)
    results.groupby(["method", "modality"])["accuracy"].agg(["mean", "std"]).reset_index().to_csv(
        results_dir / "speaker_cv_summary.csv", index=False
    )
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
