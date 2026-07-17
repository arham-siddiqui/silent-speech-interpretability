#!/usr/bin/env python3
"""Run true encoder-disjoint CV from fold-specific embedding artifacts."""

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
    repetition_training_arrays,
    validate_pair_labels,
)
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.data.synthetic import MODALITIES
from silent_speech_interpretability.evals.metrics import accuracy, macro_f1
from silent_speech_interpretability.evals.true_cv import configured_fold_embedding_paths, metadata_path_for_fold, missing_embedding_paths
from silent_speech_interpretability.models.fusion import (
    PrototypeClassifier,
    borda_count_fusion,
    consistency_weighted_fusion,
    equal_weight_fusion,
    static_weight_fusion,
)


def _align_probs(classifier: PrototypeClassifier, embeddings: np.ndarray, class_axis: np.ndarray) -> np.ndarray:
    raw = classifier.predict_proba(embeddings)
    aligned = np.zeros((raw.shape[0], len(class_axis)), dtype=np.float32)
    class_to_column = {int(cls): i for i, cls in enumerate(class_axis)}
    for raw_column, cls in enumerate(classifier.classes_):
        aligned[:, class_to_column[int(cls)]] = raw[:, raw_column]
    return aligned / (aligned.sum(axis=1, keepdims=True) + 1e-8)


def _load_metadata(config: dict, fold_id: int) -> dict:
    embeddings_dir = config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv")
    path = metadata_path_for_fold(embeddings_dir, fold_id)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_fold_metadata(config: dict, fold: dict, metadata: dict) -> None:
    if not metadata:
        if config.get("true_encoder_cv", {}).get("require_metadata", True):
            raise RuntimeError(f"Missing metadata for fold {fold['fold']}; run scripts/07_prepare_true_encoder_cv.py first.")
        return
    if metadata.get("status") != "completed":
        raise RuntimeError(f"Fold {fold['fold']} metadata status is {metadata.get('status')!r}, expected 'completed'.")
    if set(map(int, metadata.get("train_speakers", []))) & set(map(int, fold["test_speakers"])):
        raise RuntimeError(f"Fold {fold['fold']} encoder train speakers overlap test speakers.")


def _predictions_for(probabilities: np.ndarray, class_axis: np.ndarray) -> np.ndarray:
    return class_axis[np.argmax(probabilities, axis=1)]


def _record_predictions(
    prediction_rows: list[dict[str, object]],
    fold_id: int,
    method: str,
    modality: str,
    pairs: list[tuple[str, str]],
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    for pair, expected, predicted in zip(pairs, y_true, y_pred, strict=True):
        prediction_rows.append(
            {
                "fold": fold_id,
                "method": method,
                "modality": modality,
                "user_id": pair[0],
                "group_name": pair[1],
                "class_id": int(expected),
                "predicted_class_id": int(predicted),
                "correct": bool(int(expected) == int(predicted)),
            }
        )


def _validation_reliability_weights(
    val_scores: dict[str, float],
    floor: float = 0.05,
) -> dict[str, float]:
    if not val_scores:
        return {}
    raw = {modality: max(float(score), floor) for modality, score in val_scores.items()}
    total = sum(raw.values())
    if total <= 0.0:
        return {modality: 1.0 / len(raw) for modality in raw}
    return {modality: weight / total for modality, weight in raw.items()}


def _invalid_artifacts(config: dict, fold_id: int, metadata: dict, paths: dict[str, Path]) -> list[dict[str, object]]:
    invalid = []
    true_cv = config.get("true_encoder_cv", {})
    epoch_gates = {
        "lip": int(true_cv.get("min_lip_epochs", 0)),
        "mouth": int(true_cv.get("min_mouth_epochs", 0)),
        "uwb": int(true_cv.get("min_uwb_epochs", 0)),
        "mmwave": int(true_cv.get("min_mmwave_epochs", 0)),
        "laser": int(true_cv.get("min_laser_epochs", 0)),
    }
    for modality, min_epochs in epoch_gates.items():
        if modality not in paths or not Path(paths[modality]).exists() or min_epochs <= 0:
            continue
        training = metadata.get(f"{modality}_training", {}) if metadata else {}
        epochs = int(training.get("max_epochs", 0))
        if epochs < min_epochs:
            invalid.append(
                {
                    "fold": fold_id,
                    "modality": modality,
                    "path": str(paths[modality]),
                    "reason": f"{modality} artifact has max_epochs={epochs}, expected at least {min_epochs}",
                }
            )
    return invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--allow-missing", action="store_true", help="Write a missing-artifact report instead of failing.")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    seed_paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    results_dir = Path(config["data"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    all_missing = []
    all_invalid = []
    rows = []
    prediction_rows = []
    weight_rows = []
    class_axis = np.arange(int(config["classes"]["num_classes"]))
    fusion_config = config["fusion"]
    fusion_methods = [method for method in fusion_config["methods"] if method != "learned_gate"]
    excluded_fusion_modalities = set(fusion_config.get("excluded_modalities", []))

    for fold in folds:
        fold_id = int(fold["fold"])
        paths = configured_fold_embedding_paths(config.get("true_encoder_cv", {}), fold_id)
        missing = missing_embedding_paths(paths)
        metadata = _load_metadata(config, fold_id)
        if missing:
            all_missing.extend(
                {"fold": fold_id, "modality": modality, "path": path, "reason": "missing file"}
                for modality, path in missing.items()
            )
        invalid = _invalid_artifacts(config, fold_id, metadata, paths)
        if invalid:
            all_invalid.extend(invalid)
        if missing or invalid:
            continue
        _validate_fold_metadata(config, fold, metadata)

        payloads = {modality: load_embedding_repetitions(path) for modality, path in paths.items()}
        modalities = [modality for modality in MODALITIES if modality in payloads]
        common_payloads = {modality: payloads[modality] for modality in modalities}
        train_pairs = common_pairs(common_payloads, fold["train_speakers"])
        val_pairs = common_pairs(common_payloads, fold["val_speakers"])
        test_pairs = common_pairs(common_payloads, fold["test_speakers"])
        y_true = validate_pair_labels(common_payloads, test_pairs)

        probabilities = {}
        validation_scores = {}
        for modality in modalities:
            train_x, train_y = repetition_training_arrays(payloads[modality], train_pairs)
            test_x, _ = mean_eval_arrays(payloads[modality], test_pairs)
            classifier = PrototypeClassifier(config["fusion"]["temperature"]).fit(train_x, train_y)
            probs = _align_probs(classifier, test_x, class_axis)
            predictions = _predictions_for(probs, class_axis)
            probabilities[modality] = probs

            if val_pairs:
                val_x, val_y = mean_eval_arrays(payloads[modality], val_pairs)
                val_probs = _align_probs(classifier, val_x, class_axis)
                validation_scores[modality] = accuracy(val_y, _predictions_for(val_probs, class_axis))

            rows.append(
                {
                    "fold": fold_id,
                    "method": "prototype",
                    "modality": modality,
                    "accuracy": accuracy(y_true, predictions),
                    "macro_f1": macro_f1(y_true, predictions),
                    "num_train": len(train_pairs),
                    "num_test": len(test_pairs),
                    "encoder_disjoint_test": True,
                }
            )
            _record_predictions(prediction_rows, fold_id, "prototype", modality, test_pairs, y_true, predictions)

        reliability_weights = _validation_reliability_weights(validation_scores)
        fusion_probabilities = {k: v for k, v in probabilities.items() if k not in excluded_fusion_modalities}
        if not fusion_probabilities:
            raise RuntimeError("No modalities remain after applying fusion.excluded_modalities.")
        fusion_validation_scores = {
            modality: score for modality, score in validation_scores.items() if modality in fusion_probabilities
        }
        reliability_weights = _validation_reliability_weights(fusion_validation_scores)
        for modality in modalities:
            weight_rows.append(
                {
                    "fold": fold_id,
                    "method": "validation_weighted",
                    "modality": modality,
                    "validation_accuracy": validation_scores.get(modality, np.nan),
                    "weight": reliability_weights.get(modality, 0.0),
                    "included_in_fusion": modality in fusion_probabilities,
                    "num_val": len(val_pairs),
                }
            )

        for method in fusion_methods:
            if method == "equal_weight":
                fused = equal_weight_fusion(fusion_probabilities)
            elif method == "equal_weight_no_mouth":
                fused = equal_weight_fusion({k: v for k, v in probabilities.items() if k != "mouth"})
            elif method == "borda":
                fused = borda_count_fusion(fusion_probabilities)
            elif method == "consistency_weighted":
                fused, _weights = consistency_weighted_fusion(fusion_probabilities)
            elif method == "validation_weighted":
                fused = static_weight_fusion(fusion_probabilities, reliability_weights)
            else:
                continue
            predictions = _predictions_for(fused, class_axis)
            rows.append(
                {
                    "fold": fold_id,
                    "method": method,
                    "modality": "fusion",
                    "accuracy": accuracy(y_true, predictions),
                    "macro_f1": macro_f1(y_true, predictions),
                    "num_train": len(train_pairs),
                    "num_test": len(test_pairs),
                    "encoder_disjoint_test": True,
                }
            )
            _record_predictions(prediction_rows, fold_id, method, "fusion", test_pairs, y_true, predictions)

    missing_or_invalid = all_missing + all_invalid
    if missing_or_invalid:
        missing_df = pd.DataFrame(missing_or_invalid)
        missing_df.to_csv(results_dir / "true_encoder_cv_missing_artifacts.csv", index=False)
        message = (
            f"Missing or invalid {len(missing_or_invalid)} fold-specific embedding artifacts. "
            f"See {results_dir / 'true_encoder_cv_missing_artifacts.csv'}"
        )
        if args.allow_missing:
            print(message)
            return
        raise RuntimeError(message)
    (results_dir / "true_encoder_cv_missing_artifacts.csv").write_text("fold,modality,path,reason\n", encoding="utf-8")

    results = pd.DataFrame(rows)
    results.to_csv(results_dir / "true_encoder_cv_results.csv", index=False)
    results.groupby(["method", "modality"])["accuracy"].agg(["mean", "std", "count"]).reset_index().to_csv(
        results_dir / "true_encoder_cv_summary.csv", index=False
    )
    predictions = pd.DataFrame(prediction_rows)
    predictions.to_csv(results_dir / "true_encoder_cv_predictions.csv", index=False)
    predictions.groupby(["method", "modality", "class_id"])["correct"].agg(["mean", "sum", "count"]).reset_index().rename(
        columns={"mean": "accuracy", "sum": "num_correct", "count": "num_samples"}
    ).to_csv(results_dir / "true_encoder_cv_per_class.csv", index=False)
    pd.DataFrame(weight_rows).to_csv(results_dir / "true_encoder_cv_fusion_weights.csv", index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
