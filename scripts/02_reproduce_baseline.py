#!/usr/bin/env python3
"""Reproduce prototype-fusion baselines from existing or synthetic embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import discover_embedding_paths
from silent_speech_interpretability.data.synthetic import MODALITIES, make_synthetic_embeddings, make_synthetic_manifest
from silent_speech_interpretability.evals.metrics import accuracy, bootstrap_accuracy_ci, macro_f1
from silent_speech_interpretability.models.fusion import (
    PrototypeClassifier,
    borda_count_fusion,
    consistency_weighted_fusion,
    equal_weight_fusion,
)


def _load_modality(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {key: data[key] for key in ("embeddings", "labels", "user_ids", "group_names")}


def _ensure_embeddings(config: dict) -> dict[str, Path]:
    paths = discover_embedding_paths([config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    if paths:
        return paths
    manifest = make_synthetic_manifest()
    return make_synthetic_embeddings(config["data"]["embeddings_dir"], manifest)


def _split_indices(user_ids: np.ndarray, speakers: list[int]) -> np.ndarray:
    return np.flatnonzero(np.isin(user_ids.astype(int), speakers))


def _evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {"accuracy": accuracy(y_true, y_pred), "macro_f1": macro_f1(y_true, y_pred)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    results_dir = Path(config["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    paths = _ensure_embeddings(config)
    methods = [m for m in config["fusion"]["methods"] if m != "learned_gate"]

    per_modality_rows = []
    probabilities = {}
    y_true_ref = None
    user_ids_ref = None

    for modality in MODALITIES:
        if modality not in paths:
            continue
        payload = _load_modality(paths[modality])
        train_idx = _split_indices(payload["user_ids"], config["splits"]["fixed_train_speakers"])
        test_idx = _split_indices(payload["user_ids"], config["splits"]["fixed_test_speakers"])
        clf = PrototypeClassifier(config["fusion"]["temperature"]).fit(payload["embeddings"][train_idx], payload["labels"][train_idx])
        probs = clf.predict_proba(payload["embeddings"][test_idx])
        preds = clf.classes_[np.argmax(probs, axis=1)]
        y_true = payload["labels"][test_idx]
        probabilities[modality] = probs
        y_true_ref = y_true if y_true_ref is None else y_true_ref
        user_ids_ref = payload["user_ids"][test_idx] if user_ids_ref is None else user_ids_ref
        row = {"method": "prototype", "modality": modality, "num_train": len(train_idx), "num_test": len(test_idx)}
        row.update(_evaluate_predictions(y_true, preds))
        per_modality_rows.append(row)

    fusion_rows = []
    ci_payload = {}
    if probabilities and y_true_ref is not None:
        for method in methods:
            if method == "equal_weight":
                fused = equal_weight_fusion(probabilities)
            elif method == "borda":
                fused = borda_count_fusion(probabilities)
            elif method == "consistency_weighted":
                fused, _weights = consistency_weighted_fusion(probabilities)
            else:
                continue
            preds = np.argmax(fused, axis=1)
            row = {"method": method, "modality": "fusion", "num_train": None, "num_test": len(y_true_ref)}
            row.update(_evaluate_predictions(y_true_ref, preds))
            fusion_rows.append(row)
            ci_payload[method] = bootstrap_accuracy_ci(y_true_ref, preds, n_boot=300)

    pd.DataFrame(per_modality_rows).to_csv(results_dir / "per_modality_results.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(results_dir / "fixed_split_results.csv", index=False)
    (results_dir / "bootstrap_ci.json").write_text(json.dumps(ci_payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(per_modality_rows + fusion_rows).to_string(index=False))


if __name__ == "__main__":
    main()
