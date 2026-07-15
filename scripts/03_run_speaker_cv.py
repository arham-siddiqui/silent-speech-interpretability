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
from silent_speech_interpretability.data.embeddings import (
    common_pairs,
    load_embedding_repetitions,
    mean_eval_arrays,
    repetition_training_arrays,
    validate_pair_labels,
)
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.synthetic import MODALITIES
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits, save_split_json
from silent_speech_interpretability.evals.metrics import accuracy, macro_f1
from silent_speech_interpretability.models.fusion import (
    PrototypeClassifier,
    borda_count_fusion,
    consistency_weighted_fusion,
    equal_weight_fusion,
)


def _ensure_embeddings(config: dict) -> dict[str, Path]:
    paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    return paths


def _enabled_modalities(config: dict, paths: dict[str, Path]) -> list[str]:
    enabled = []
    for modality in MODALITIES:
        if modality in paths and config["modalities"].get(modality, {}).get("enabled", True):
            enabled.append(modality)
    return enabled


def _probs_on_class_axis(classifier: PrototypeClassifier, embeddings: np.ndarray, class_axis: np.ndarray) -> np.ndarray:
    raw = classifier.predict_proba(embeddings)
    aligned = np.zeros((raw.shape[0], len(class_axis)), dtype=np.float32)
    class_to_column = {int(cls): i for i, cls in enumerate(class_axis)}
    for raw_column, cls in enumerate(classifier.classes_):
        if int(cls) in class_to_column:
            aligned[:, class_to_column[int(cls)]] = raw[:, raw_column]
    row_sums = aligned.sum(axis=1, keepdims=True)
    return aligned / (row_sums + 1e-8)


def _metric_row(
    fold: dict,
    method: str,
    modality: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_train: int,
    num_val: int,
    encoder_train_speakers: set[int],
) -> dict[str, object]:
    test_speakers = {int(speaker) for speaker in fold["test_speakers"]}
    seen_test_speakers = sorted(test_speakers & encoder_train_speakers)
    return {
        "fold": fold["fold"],
        "method": method,
        "modality": modality,
        "accuracy": accuracy(y_true, y_pred),
        "macro_f1": macro_f1(y_true, y_pred),
        "num_train": num_train,
        "num_val": num_val,
        "num_test": len(y_true),
        "test_speakers": ",".join(map(str, fold["test_speakers"])),
        "encoder_seen_test_speakers": ",".join(map(str, seen_test_speakers)),
        "num_encoder_seen_test_speakers": len(seen_test_speakers),
        "encoder_disjoint_test": len(seen_test_speakers) == 0,
    }


def _write_plots(results: pd.DataFrame, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional plotting dependency
        print(f"Skipping CV plots because matplotlib is unavailable: {exc}")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    summary = results.groupby(["method", "modality"], as_index=False)["accuracy"].agg(["mean", "std"]).reset_index()
    summary["label"] = summary.apply(
        lambda row: row["modality"] if row["method"] == "prototype" else row["method"],
        axis=1,
    )
    summary = summary.sort_values("mean", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(summary["label"], summary["mean"], yerr=summary["std"].fillna(0), capsize=3)
    plt.ylabel("Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "speaker_cv_accuracy.png", dpi=180)
    plt.close()

    modality = results[results["method"].eq("prototype")]
    if not modality.empty:
        pivot = modality.pivot(index="fold", columns="modality", values="accuracy").sort_index()
        pivot.plot(marker="o", figsize=(10, 5))
        plt.ylabel("Accuracy")
        plt.ylim(0, 1)
        plt.tight_layout()
        plt.savefig(figures_dir / "speaker_cv_by_modality.png", dpi=180)
        plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    paths = _ensure_embeddings(config)
    payloads = {modality: load_embedding_repetitions(path) for modality, path in paths.items()}
    modalities = _enabled_modalities(config, paths)
    if not modalities:
        raise RuntimeError("No enabled embedding modalities were found.")

    manifest = build_manifest(paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    save_split_json(folds, "artifacts/splits/speaker_kfold_5.json")

    labels_union = sorted({int(label) for payload in payloads.values() for label in payload["labels"].values()})
    configured_classes = int(config["classes"]["num_classes"])
    class_axis = np.array(sorted(set(range(configured_classes)) | set(labels_union)), dtype=np.int64)
    fusion_methods = [method for method in config["fusion"]["methods"] if method != "learned_gate"]
    encoder_train_speakers = {int(speaker) for speaker in config.get("evaluation", {}).get("embedding_encoder_train_speakers", [])}
    rows = []

    for fold in folds:
        common_payloads = {modality: payloads[modality] for modality in modalities}
        train_pairs = common_pairs(common_payloads, fold["train_speakers"])
        val_pairs = common_pairs(common_payloads, fold["val_speakers"])
        test_pairs = common_pairs(common_payloads, fold["test_speakers"])
        if not train_pairs or not test_pairs:
            print(f"Skipping fold {fold['fold']} because strict multimodal intersection is empty.")
            continue

        y_true = validate_pair_labels(common_payloads, test_pairs)
        probabilities = {}
        for modality in modalities:
            payload = payloads[modality]
            train_x, train_y = repetition_training_arrays(payload, train_pairs)
            test_x, _ = mean_eval_arrays(payload, test_pairs)
            classifier = PrototypeClassifier(config["fusion"]["temperature"]).fit(
                train_x,
                train_y,
            )
            probs = _probs_on_class_axis(classifier, test_x, class_axis)
            predictions = class_axis[np.argmax(probs, axis=1)]
            probabilities[modality] = probs
            rows.append(
                _metric_row(
                    fold,
                    method="prototype",
                    modality=modality,
                    y_true=y_true,
                    y_pred=predictions,
                    num_train=len(train_pairs),
                    num_val=len(val_pairs),
                    encoder_train_speakers=encoder_train_speakers,
                )
            )

        for method in fusion_methods:
            if method == "equal_weight":
                fused = equal_weight_fusion(probabilities)
            elif method == "borda":
                fused = borda_count_fusion(probabilities)
            elif method == "consistency_weighted":
                fused, _weights = consistency_weighted_fusion(probabilities)
            else:
                continue
            predictions = class_axis[np.argmax(fused, axis=1)]
            rows.append(
                _metric_row(
                    fold,
                    method=method,
                    modality="fusion",
                    y_true=y_true,
                    y_pred=predictions,
                    num_train=len(train_pairs),
                    num_val=len(val_pairs),
                    encoder_train_speakers=encoder_train_speakers,
                )
            )

    results = pd.DataFrame(rows)
    results_dir = Path(config["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_dir / "speaker_cv_results.csv", index=False)
    results.groupby(["method", "modality"])["accuracy"].agg(["mean", "std", "count"]).reset_index().to_csv(
        results_dir / "speaker_cv_summary.csv", index=False
    )
    _write_plots(results, Path(config["data"]["figures_dir"]))
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
