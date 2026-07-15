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
from silent_speech_interpretability.data.embeddings import (
    common_pairs,
    load_embedding_repetitions,
    mean_eval_arrays,
    modality_pairs,
    repetition_training_arrays,
    validate_pair_labels,
)
from silent_speech_interpretability.data.manifest import resolve_embedding_paths
from silent_speech_interpretability.data.synthetic import MODALITIES
from silent_speech_interpretability.evals.metrics import accuracy, bootstrap_accuracy_ci, macro_f1
from silent_speech_interpretability.models.fusion import (
    PrototypeClassifier,
    borda_count_fusion,
    consistency_weighted_fusion,
    equal_weight_fusion,
)


def _ensure_embeddings(config: dict) -> dict[str, Path]:
    paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    return paths


def _probs_on_class_axis(classifier: PrototypeClassifier, embeddings: np.ndarray, class_axis: np.ndarray) -> np.ndarray:
    raw = classifier.predict_proba(embeddings)
    aligned = np.zeros((raw.shape[0], len(class_axis)), dtype=np.float32)
    class_to_column = {int(cls): i for i, cls in enumerate(class_axis)}
    for raw_column, cls in enumerate(classifier.classes_):
        if int(cls) in class_to_column:
            aligned[:, class_to_column[int(cls)]] = raw[:, raw_column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return aligned / (row_sums + 1e-8)


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
    payloads = {modality: load_embedding_repetitions(path) for modality, path in paths.items()}
    modalities = [modality for modality in MODALITIES if modality in payloads]
    labels_union = sorted({int(label) for payload in payloads.values() for label in payload["labels"].values()})
    class_axis = np.array(sorted(set(range(int(config["classes"]["num_classes"]))) | set(labels_union)), dtype=np.int64)

    per_modality_rows = []
    prediction_rows = []

    for modality in modalities:
        payload = payloads[modality]
        train_pairs = modality_pairs(payload, config["splits"]["fixed_train_speakers"])
        test_pairs = modality_pairs(payload, config["splits"]["fixed_test_speakers"])
        train_x, train_y = repetition_training_arrays(payload, train_pairs)
        test_x, y_true = mean_eval_arrays(payload, test_pairs)
        clf = PrototypeClassifier(config["fusion"]["temperature"]).fit(train_x, train_y)
        probs = clf.predict_proba(test_x)
        preds = clf.classes_[np.argmax(probs, axis=1)]
        row = {"method": "prototype", "modality": modality, "num_train": len(train_x), "num_test": len(test_pairs)}
        row.update(_evaluate_predictions(y_true, preds))
        per_modality_rows.append(row)
        prediction_rows.extend(
            {
                "method": "prototype",
                "modality": modality,
                "sample_id": f"{pair[0]}::{pair[1]}",
                "y_true": int(true),
                "y_pred": int(pred),
            }
            for pair, true, pred in zip(test_pairs, y_true, preds)
        )

    fusion_rows = []
    ci_payload = {}
    if modalities:
        common_payloads = {modality: payloads[modality] for modality in modalities}
        train_pairs = common_pairs(common_payloads, config["splits"]["fixed_train_speakers"])
        test_pairs = common_pairs(common_payloads, config["splits"]["fixed_test_speakers"])
        probabilities = {}
        y_true_ref = validate_pair_labels(common_payloads, test_pairs)
        for modality in modalities:
            payload = payloads[modality]
            train_x, train_y = repetition_training_arrays(payload, train_pairs)
            test_x, _ = mean_eval_arrays(payload, test_pairs)
            clf = PrototypeClassifier(config["fusion"]["temperature"]).fit(train_x, train_y)
            probabilities[modality] = _probs_on_class_axis(clf, test_x, class_axis)

        for method in methods:
            if method == "equal_weight":
                fused = equal_weight_fusion(probabilities)
            elif method == "borda":
                fused = borda_count_fusion(probabilities)
            elif method == "consistency_weighted":
                fused, _weights = consistency_weighted_fusion(probabilities)
            else:
                continue
            preds = class_axis[np.argmax(fused, axis=1)]
            row = {"method": method, "modality": "fusion", "num_train": len(train_pairs), "num_test": len(test_pairs)}
            row.update(_evaluate_predictions(y_true_ref, preds))
            fusion_rows.append(row)
            ci_payload[method] = bootstrap_accuracy_ci(y_true_ref, preds, n_boot=300)
            prediction_rows.extend(
                {
                    "method": method,
                    "modality": "fusion",
                    "sample_id": f"{pair[0]}::{pair[1]}",
                    "y_true": int(true),
                    "y_pred": int(pred),
                }
                for pair, true, pred in zip(test_pairs, y_true_ref, preds)
            )

    pd.DataFrame(per_modality_rows).to_csv(results_dir / "per_modality_results.csv", index=False)
    pd.DataFrame(fusion_rows).to_csv(results_dir / "fixed_split_results.csv", index=False)
    pd.DataFrame(prediction_rows).to_csv(results_dir / "fixed_split_predictions.csv", index=False)
    (results_dir / "bootstrap_ci.json").write_text(json.dumps(ci_payload, indent=2), encoding="utf-8")
    print(pd.DataFrame(per_modality_rows + fusion_rows).to_string(index=False))


if __name__ == "__main__":
    main()
