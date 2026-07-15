#!/usr/bin/env python3
"""Compare auditable baselines with original fusionGate-style prototype fusion."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.embeddings import (
    common_pairs,
    load_embedding_repetitions,
    stacked_mean_eval_arrays,
    stacked_repetition_training_arrays,
    validate_pair_labels,
)
from silent_speech_interpretability.data.manifest import resolve_embedding_paths
from silent_speech_interpretability.data.synthetic import MODALITIES
from silent_speech_interpretability.evals.metrics import accuracy, macro_f1
from silent_speech_interpretability.models.fusion.legacy import (
    compute_stacked_prototypes,
    legacy_borda_predictions,
    legacy_equal_weight_predictions,
    legacy_hard_consistency_predictions,
)


LEGACY_METHODS = {
    "legacy_equal_weight": legacy_equal_weight_predictions,
    "legacy_borda": legacy_borda_predictions,
    "legacy_hard_consistency": legacy_hard_consistency_predictions,
}


def _metric_row(method: str, y_true, y_pred, num_train: int, num_val: int, num_test: int) -> dict[str, object]:
    return {
        "baseline_family": "legacy_compatible",
        "method": method,
        "modality": "fusion",
        "num_train": num_train,
        "num_val": num_val,
        "num_test": num_test,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    payloads = {modality: load_embedding_repetitions(path) for modality, path in paths.items()}
    modalities = [modality for modality in MODALITIES if modality in payloads]
    common_payloads = {modality: payloads[modality] for modality in modalities}

    train_pairs = common_pairs(common_payloads, config["splits"]["fixed_train_speakers"])
    val_pairs = common_pairs(common_payloads, config["splits"]["fixed_val_speakers"])
    test_pairs = common_pairs(common_payloads, config["splits"]["fixed_test_speakers"])
    validate_pair_labels(common_payloads, train_pairs + val_pairs + test_pairs)

    train_x, train_y, _train_users = stacked_repetition_training_arrays(common_payloads, modalities, train_pairs)
    test_x, test_y = stacked_mean_eval_arrays(common_payloads, modalities, test_pairs)
    num_classes = int(config["classes"]["num_classes"])
    prototypes = compute_stacked_prototypes(train_x, train_y, num_classes)

    rows = []
    for method, predict_fn in LEGACY_METHODS.items():
        preds = predict_fn(test_x, prototypes, temp=10.0)
        rows.append(_metric_row(method, test_y, preds, len(train_x), len(val_pairs), len(test_pairs)))

    results_dir = Path(config["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    legacy = pd.DataFrame(rows)
    legacy.to_csv(results_dir / "legacy_compatible_fixed_split_results.csv", index=False)

    comparison = legacy.copy()
    comparison["source"] = "legacy_compatible"
    fixed_path = results_dir / "fixed_split_results.csv"
    if fixed_path.exists():
        current = pd.read_csv(fixed_path)
        current = current[current["modality"].eq("fusion")].copy()
        current["baseline_family"] = "auditable_current"
        current["num_val"] = len(val_pairs)
        current["source"] = "auditable_current"
        comparison = pd.concat(
            [
                current[
                    [
                        "baseline_family",
                        "method",
                        "modality",
                        "num_train",
                        "num_val",
                        "num_test",
                        "accuracy",
                        "macro_f1",
                        "source",
                    ]
                ],
                comparison[
                    [
                        "baseline_family",
                        "method",
                        "modality",
                        "num_train",
                        "num_val",
                        "num_test",
                        "accuracy",
                        "macro_f1",
                        "source",
                    ]
                ],
            ],
            ignore_index=True,
        )

    comparison.to_csv(results_dir / "baseline_comparison.csv", index=False)
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
