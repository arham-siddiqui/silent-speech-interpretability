#!/usr/bin/env python3
"""Probe temporal sensor states for measured lip articulation trajectories."""

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


TARGET_NAMES = ("lip_aperture", "lip_width", "lip_motion")


def _index(data, prefix: str) -> dict[tuple[str, str], int]:
    return {
        (str(user), str(group)): index
        for index, (user, group) in enumerate(zip(data[f"{prefix}_user_ids"], data[f"{prefix}_group_names"], strict=True))
    }


def _pair_arrays(data, modalities: list[str], speakers: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    prefixes = modalities + ["articulation"]
    indices = {prefix: _index(data, prefix) for prefix in prefixes}
    speaker_set = {str(speaker) for speaker in speakers}
    pairs = set(indices[prefixes[0]])
    for prefix in prefixes[1:]:
        pairs &= set(indices[prefix])
    pairs = sorted(pair for pair in pairs if pair[0] in speaker_set)
    x = np.concatenate(
        [data[f"{modality}_values"][[indices[modality][pair] for pair in pairs]] for modality in modalities],
        axis=2,
    ).astype(np.float32)
    y = data["articulation_values"][[indices["articulation"][pair] for pair in pairs]].astype(np.float32)
    labels = data["articulation_labels"][[indices["articulation"][pair] for pair in pairs]].astype(np.int64)
    return x, y, labels, pairs


def _class_position_means(y: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(label): y[labels == label].mean(axis=0) for label in np.unique(labels)}


def _baseline(means: dict[int, np.ndarray], labels: np.ndarray) -> np.ndarray:
    return np.stack([means[int(label)] for label in labels])


def _r2(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray([r2_score(y[:, :, index].reshape(-1), prediction[:, :, index].reshape(-1)) for index in range(y.shape[2])])


def _correlation(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    values = []
    for index in range(y.shape[2]):
        a = y[:, :, index].reshape(-1)
        b = prediction[:, :, index].reshape(-1)
        values.append(float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-8 and b.std() > 1e-8 else 0.0)
    return np.asarray(values)


def _fit_probe(
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    test: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray, np.ndarray]:
    train_x, train_y, train_labels = train
    val_x, val_y, val_labels = val
    test_x, test_y, test_labels = test
    class_means = _class_position_means(train_y, train_labels)
    train_base = _baseline(class_means, train_labels)
    val_base = _baseline(class_means, val_labels)
    test_base = _baseline(class_means, test_labels)
    train_residual = train_y - train_base
    best_score, best_alpha = -np.inf, 1.0
    # Accelerate can emit spurious overflow warnings for finite float32 matmuls on
    # macOS. The finite-value assertions below retain the actual safety check.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for alpha in (0.1, 1.0, 10.0, 100.0, 1000.0):
            model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="lsqr", tol=1e-6))
            model.fit(train_x.reshape(-1, train_x.shape[2]), train_residual.reshape(-1, train_y.shape[2]))
            prediction = val_base + model.predict(val_x.reshape(-1, val_x.shape[2])).reshape(val_y.shape)
            if not np.isfinite(prediction).all():
                raise ValueError(f"Non-finite validation prediction for alpha={alpha}")
            score = float(_r2(val_y, prediction).mean())
            if score > best_score:
                best_score, best_alpha = score, alpha
    model = make_pipeline(StandardScaler(), Ridge(alpha=best_alpha, solver="lsqr", tol=1e-6))
    combined_x = np.concatenate([train_x, val_x])
    combined_y = np.concatenate([train_y, val_y])
    combined_labels = np.concatenate([train_labels, val_labels])
    combined_means = _class_position_means(combined_y, combined_labels)
    combined_base = _baseline(combined_means, combined_labels)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(combined_x.reshape(-1, combined_x.shape[2]), (combined_y - combined_base).reshape(-1, combined_y.shape[2]))
        test_class_base = _baseline(combined_means, test_labels)
        prediction = test_class_base + model.predict(test_x.reshape(-1, test_x.shape[2])).reshape(test_y.shape)
    if not np.isfinite(prediction).all():
        raise ValueError("Non-finite test prediction")
    position_mean = combined_y.mean(axis=0, keepdims=True)
    position_prediction = np.repeat(position_mean, len(test_y), axis=0)
    return prediction, test_class_base, best_alpha, position_prediction, test_y


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
        input_dim=int(checkpoint["input_dim"]),
        target_dim=int(checkpoint["target_dim"]),
        hidden_dim=int(checkpoint.get("hidden_dim", 256)),
        bottleneck_dim=int(checkpoint.get("bottleneck_dim", 64)),
        num_classes=int(checkpoint.get("num_classes", 30)),
        num_segments=int(checkpoint["num_segments"]),
        **extra,
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalized = (x - np.asarray(checkpoint["input_mean"])) / np.asarray(checkpoint["input_std"])
    with torch.no_grad():
        return model(torch.tensor(normalized, dtype=torch.float32))["bottleneck"].numpy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--activations-dir", default="artifacts/activations/temporal_sensors")
    parser.add_argument("--student-dir", default="artifacts/students/temporal_sensor_cv")
    parser.add_argument("--multitask-student-dir", default="artifacts/students/temporal_sensor_multitask_cv")
    parser.add_argument("--attention-student-dir", default="artifacts/students/temporal_sensor_attention_cv")
    parser.add_argument("--output", default="reports/results/temporal_articulation_probe_results.csv")
    parser.add_argument("--summary-output", default="reports/results/temporal_articulation_probe_summary.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_paths, _ = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    variants = {
        "lip": ["lip"],
        "laser": ["laser"],
        "mmwave": ["mmwave"],
        "uwb": ["uwb"],
        "contactless_nonlip": ["laser", "mmwave", "uwb"],
        "all_modalities": ["lip", "laser", "mmwave", "uwb"],
    }
    rows = []
    for fold in folds:
        fold_id = int(fold["fold"])
        data = np.load(Path(args.activations_dir) / f"fold_{fold_id}_temporal_sensors.npz")
        for variant, modalities in variants.items():
            raw = {
                split: _pair_arrays(data, modalities, fold[f"{split}_speakers"])
                for split in ("train", "val", "test")
            }
            representations = {split: raw[split][0] for split in raw}
            student_variants: list[tuple[str, dict[str, np.ndarray] | None]] = []
            if variant == "all_modalities":
                student_variants.append(
                    (
                        "temporal_student",
                        {
                            split: _student_representation(
                                Path(args.student_dir) / f"fold_{fold_id}_temporal_sensor_student.pt",
                                raw[split][0],
                            )
                            for split in raw
                        },
                    )
                )
                multitask_path = Path(args.multitask_student_dir) / f"fold_{fold_id}_temporal_sensor_multitask.pt"
                if multitask_path.exists():
                    student_variants.append(
                        (
                            "multitask_temporal_student",
                            {split: _student_representation(multitask_path, raw[split][0]) for split in raw},
                        )
                    )
                attention_path = Path(args.attention_student_dir) / f"fold_{fold_id}_temporal_sensor_attention.pt"
                if attention_path.exists():
                    student_variants.append(
                        (
                            "attention_temporal_student",
                            {split: _student_representation(attention_path, raw[split][0]) for split in raw},
                        )
                    )
            for representation_name, values in [(variant, representations), *student_variants]:
                if values is None:
                    continue
                prepared = {
                    split: (values[split], raw[split][1], raw[split][2])
                    for split in raw
                }
                prediction, class_base, alpha, position_prediction, test_y = _fit_probe(
                    prepared["train"], prepared["val"], prepared["test"]
                )
                model_r2 = _r2(test_y, prediction)
                class_r2 = _r2(test_y, class_base)
                position_r2 = _r2(test_y, position_prediction)
                correlation = _correlation(test_y, prediction)
                reversed_correlation = _correlation(test_y[:, ::-1], prediction)
                for target_index, target in enumerate(TARGET_NAMES):
                    rows.append(
                        {
                            "fold": fold_id,
                            "representation": representation_name,
                            "target": target,
                            "alpha": alpha,
                            "r2": model_r2[target_index],
                            "class_position_baseline_r2": class_r2[target_index],
                            "position_baseline_r2": position_r2[target_index],
                            "delta_r2_vs_class_position": model_r2[target_index] - class_r2[target_index],
                            "correlation": correlation[target_index],
                            "reversed_correlation": reversed_correlation[target_index],
                            "order_correlation_margin": correlation[target_index] - reversed_correlation[target_index],
                            "num_test": len(test_y),
                        }
                    )
        print(f"ARTICULATION_PROGRESS fold={fold_id}", flush=True)

    results = pd.DataFrame(rows)
    summary = results.groupby(["representation", "target"], as_index=False).agg(
        r2_mean=("r2", "mean"),
        r2_std=("r2", "std"),
        class_position_baseline_r2_mean=("class_position_baseline_r2", "mean"),
        delta_r2_mean=("delta_r2_vs_class_position", "mean"),
        correlation_mean=("correlation", "mean"),
        order_margin_mean=("order_correlation_margin", "mean"),
        folds=("fold", "nunique"),
    )
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary.to_csv(summary_output, index=False)
    print(f"Saved temporal articulation probes to {output}")


if __name__ == "__main__":
    main()
