#!/usr/bin/env python3
"""Probe speaker-disjoint temporal sensor states for aligned phonetic occupancy."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.models.students.temporal_sensor_student import (
    ModalityAttentionTemporalStudent,
    MultitaskTemporalSensorStudent,
    TemporalSensorStudent,
)


def _index(user_ids, group_names) -> dict[tuple[str, str], int]:
    return {(str(user), str(group)): index for index, (user, group) in enumerate(zip(user_ids, group_names, strict=True))}


def _pair_arrays(data, targets, modalities: list[str], speakers: list[int], minimum_confidence: float):
    indices = {modality: _index(data[f"{modality}_user_ids"], data[f"{modality}_group_names"]) for modality in modalities}
    target_index = _index(targets["user_ids"], targets["group_names"])
    pairs = set(target_index)
    for modality in modalities:
        pairs &= set(indices[modality])
    speaker_set = {str(speaker) for speaker in speakers}
    pairs = sorted(
        pair for pair in pairs
        if pair[0] in speaker_set and float(targets["alignment_confidence"][target_index[pair]]) >= minimum_confidence
    )
    x = np.concatenate(
        [data[f"{modality}_values"][[indices[modality][pair] for pair in pairs]] for modality in modalities], axis=2
    ).astype(np.float32)
    y = targets["values"][[target_index[pair] for pair in pairs]].astype(np.float32)
    labels = targets["labels"][[target_index[pair] for pair in pairs]].astype(np.int64)
    return x, y, labels


def _class_means(y: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(label): y[labels == label].mean(axis=0) for label in np.unique(labels)}


def _baseline(means: dict[int, np.ndarray], labels: np.ndarray) -> np.ndarray:
    return np.stack([means[int(label)] for label in labels])


def _r2(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray([r2_score(y[:, :, index].reshape(-1), prediction[:, :, index].reshape(-1)) for index in range(y.shape[2])])


def _correlation(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    output = []
    for index in range(y.shape[2]):
        actual = y[:, :, index].reshape(-1)
        predicted = prediction[:, :, index].reshape(-1)
        output.append(float(np.corrcoef(actual, predicted)[0, 1]) if actual.std() > 1e-8 and predicted.std() > 1e-8 else 0.0)
    return np.asarray(output)


def _fit_probe(train, val, test):
    train_x, train_y, train_labels = train
    val_x, val_y, val_labels = val
    test_x, test_y, test_labels = test
    means = _class_means(train_y, train_labels)
    train_base = _baseline(means, train_labels)
    val_base = _baseline(means, val_labels)
    best_score, best_alpha = -np.inf, 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-6))
            model.fit(train_x.reshape(-1, train_x.shape[2]), (train_y - train_base).reshape(-1, train_y.shape[2]))
            prediction = val_base + model.predict(val_x.reshape(-1, val_x.shape[2])).reshape(val_y.shape)
            score = float(np.nanmean(_r2(val_y, prediction)))
            if score > best_score:
                best_score, best_alpha = score, alpha
    combined_x = np.concatenate([train_x, val_x])
    combined_y = np.concatenate([train_y, val_y])
    combined_labels = np.concatenate([train_labels, val_labels])
    means = _class_means(combined_y, combined_labels)
    combined_base = _baseline(means, combined_labels)
    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, solver="lsqr", tol=1e-6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(combined_x.reshape(-1, combined_x.shape[2]), (combined_y - combined_base).reshape(-1, combined_y.shape[2]))
    test_base = _baseline(means, test_labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        prediction = test_base + model.predict(test_x.reshape(-1, test_x.shape[2])).reshape(test_y.shape)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite test prediction")
    return prediction, test_base, best_alpha


def _student_representation(checkpoint_path: Path, x: np.ndarray) -> np.ndarray:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_type = checkpoint.get("model_type")
    if model_type == "modality_attention_temporal_sensor":
        model_class = ModalityAttentionTemporalStudent
    elif model_type == "multitask_temporal_sensor":
        model_class = MultitaskTemporalSensorStudent
    else:
        model_class = TemporalSensorStudent
    extra = {"num_modalities": int(checkpoint["num_modalities"])} if model_class is ModalityAttentionTemporalStudent else {}
    model = model_class(
        input_dim=int(checkpoint["input_dim"]), target_dim=int(checkpoint["target_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)), bottleneck_dim=int(checkpoint.get("bottleneck_dim", 64)),
        num_classes=int(checkpoint.get("num_classes", 30)), num_segments=int(checkpoint["num_segments"]), **extra,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalized = (x - np.asarray(checkpoint["input_mean"])) / np.asarray(checkpoint["input_std"])
    with torch.no_grad():
        return model(torch.tensor(normalized, dtype=torch.float32))["bottleneck"].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--targets", default="artifacts/forced_alignment/phonetic_segment_targets.npz")
    parser.add_argument("--activations-dir", default="artifacts/activations/temporal_sensors")
    parser.add_argument("--multitask-student-dir", default="artifacts/students/temporal_sensor_multitask_cv")
    parser.add_argument("--attention-student-dir", default="artifacts/students/temporal_sensor_attention_cv")
    parser.add_argument("--minimum-confidence", type=float, default=0.05)
    parser.add_argument("--output", default="reports/results/temporal_phonetic_probe_results.csv")
    parser.add_argument("--summary-output", default="reports/results/temporal_phonetic_probe_summary.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_paths, _ = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    targets = np.load(args.targets)
    feature_names = targets["feature_names"].astype(str)
    variants = {
        "lip": ["lip"], "laser": ["laser"], "mmwave": ["mmwave"], "uwb": ["uwb"],
        "contactless_nonlip": ["laser", "mmwave", "uwb"],
        "all_modalities": ["lip", "laser", "mmwave", "uwb"],
    }
    rows = []
    for fold in folds:
        fold_id = int(fold["fold"])
        data = np.load(Path(args.activations_dir) / f"fold_{fold_id}_temporal_sensors.npz")
        for variant, modalities in variants.items():
            raw = {split: _pair_arrays(data, targets, modalities, fold[f"{split}_speakers"], args.minimum_confidence) for split in ("train", "val", "test")}
            representations = [(variant, {split: raw[split][0] for split in raw})]
            if variant == "all_modalities":
                for name, directory, suffix in (
                    ("multitask_temporal_student", args.multitask_student_dir, "temporal_sensor_multitask"),
                    ("attention_temporal_student", args.attention_student_dir, "temporal_sensor_attention"),
                ):
                    checkpoint = Path(directory) / f"fold_{fold_id}_{suffix}.pt"
                    if checkpoint.exists():
                        representations.append((name, {split: _student_representation(checkpoint, raw[split][0]) for split in raw}))
            for representation_name, values in representations:
                prepared = {split: (values[split], raw[split][1], raw[split][2]) for split in raw}
                prediction, class_base, alpha = _fit_probe(prepared["train"], prepared["val"], prepared["test"])
                actual = prepared["test"][1]
                model_r2, base_r2 = _r2(actual, prediction), _r2(actual, class_base)
                correlation = _correlation(actual, prediction)
                reversed_correlation = _correlation(actual[:, ::-1], prediction)
                for index, feature in enumerate(feature_names):
                    rows.append(
                        {"fold": fold_id, "representation": representation_name, "feature": feature, "alpha": alpha,
                         "r2": model_r2[index], "class_position_baseline_r2": base_r2[index],
                         "delta_r2_vs_class_position": model_r2[index] - base_r2[index],
                         "correlation": correlation[index], "reversed_correlation": reversed_correlation[index],
                         "order_correlation_margin": correlation[index] - reversed_correlation[index],
                         "num_test": len(actual), "minimum_alignment_confidence": args.minimum_confidence}
                    )
        print(f"PHONETIC_PROBE progress={fold_id + 1}/{len(folds)}", flush=True)

    results = pd.DataFrame(rows)
    summary = results.groupby(["representation", "feature"], as_index=False).agg(
        r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        class_position_baseline_r2_mean=("class_position_baseline_r2", "mean"),
        delta_r2_mean=("delta_r2_vs_class_position", "mean"), correlation_mean=("correlation", "mean"),
        order_margin_mean=("order_correlation_margin", "mean"), folds=("fold", "nunique"),
        mean_test_rows=("num_test", "mean"),
    )
    output, summary_output = Path(args.output), Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary.to_csv(summary_output, index=False)
    print(f"Saved {len(results)} probe rows to {output}")


if __name__ == "__main__":
    main()
