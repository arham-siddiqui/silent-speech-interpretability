#!/usr/bin/env python3
"""Train and evaluate a student against ordered temporal HuBERT summaries."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.embeddings import load_embedding_repetitions, mean_eval_arrays
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.evals.true_cv import configured_fold_embedding_paths
from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent
from silent_speech_interpretability.models.teachers.teacher_targets import common_teacher_pairs, load_teacher_targets, teacher_arrays


def _cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-8)
    return np.sum(a * b, axis=-1)


def _evaluate_fold(config: dict, teacher: dict[str, object], fold: dict, checkpoint_path: Path) -> dict[str, float]:
    fold_id = int(fold["fold"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    modalities = list(checkpoint["modalities"])
    paths = configured_fold_embedding_paths(config.get("true_encoder_cv", {}), fold_id)
    payloads = {modality: load_embedding_repetitions(path) for modality, path in paths.items()}
    common_payloads = {modality: payloads[modality] for modality in modalities}
    pairs = common_teacher_pairs(common_payloads, teacher, fold["test_speakers"])
    inputs = np.concatenate([mean_eval_arrays(common_payloads[modality], pairs)[0] for modality in modalities], axis=1)
    targets, labels = teacher_arrays(teacher, pairs)
    targets = targets - np.asarray(checkpoint["teacher_center"])

    model = ArticulatoryStudent(
        modalities,
        embedding_dim=int(config["modalities"][modalities[0]]["embedding_dim"]),
        hidden_dim=int(config["student"]["hidden_dim"]),
        bottleneck_dim=int(config["student"]["bottleneck_dim"]),
        target_dim=int(teacher["target_dim"]),
        num_classes=int(config["classes"]["num_classes"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(inputs, dtype=torch.float32))
    predicted = output["target"].numpy()
    predicted_labels = output["logits"].argmax(dim=1).numpy()
    segments, segment_dim = teacher["target_shape"]
    predicted_segments = predicted.reshape(len(predicted), segments, segment_dim)
    target_segments = targets.reshape(len(targets), segments, segment_dim)
    true_by_segment = _cosine(predicted_segments, target_segments)
    reversed_cosine = _cosine(predicted_segments, target_segments[:, ::-1]).mean()
    shifted_cosine = _cosine(predicted_segments, np.roll(target_segments, 1, axis=1)).mean()
    pooled_cosine = _cosine(predicted_segments.mean(axis=1), target_segments.mean(axis=1)).mean()
    metrics = {
        "fold": fold_id,
        "num_test": len(pairs),
        "accuracy": float(np.mean(predicted_labels == labels)),
        "concatenated_cosine": float(_cosine(predicted, targets).mean()),
        "segment_cosine": float(true_by_segment.mean()),
        "reversed_segment_cosine": float(reversed_cosine),
        "shifted_segment_cosine": float(shifted_cosine),
        "order_margin_reversed": float(true_by_segment.mean() - reversed_cosine),
        "order_margin_shifted": float(true_by_segment.mean() - shifted_cosine),
        "pooled_from_segments_cosine": float(pooled_cosine),
    }
    training_metrics_path = checkpoint_path.with_name(checkpoint_path.name.replace(".pt", "_metrics.json"))
    training_metrics = json.loads(training_metrics_path.read_text(encoding="utf-8"))
    metrics["mean_target_baseline_cosine"] = float(training_metrics["target_mean_baseline"]["cosine_similarity"])
    for segment in range(segments):
        metrics[f"segment_{segment}_cosine"] = float(true_by_segment[:, segment].mean())
    return metrics


def _write_report(path: Path, results: pd.DataFrame, pooled: pd.DataFrame, segments: int) -> None:
    pooled_by_fold = pooled.set_index("fold")
    rows = "\n".join(
        f"| {int(row.fold)} | {100*row.accuracy:.1f}% | "
        f"{100*pooled_by_fold.loc[int(row.fold), 'test_accuracy']:.1f}% | "
        f"{row.segment_cosine:.3f} | {row.reversed_segment_cosine:.3f} | "
        f"{row.order_margin_reversed:+.3f} |"
        for row in results.itertuples(index=False)
    )
    margin = results.order_margin_reversed.mean()
    conclusion = (
        "The positive true-order margin indicates that the student recovers some ordered speech structure, not only an order-invariant utterance summary."
        if margin > 0.01
        else "The true-order margin is small, so this experiment does not yet show reliable recovery of segment order from the fixed sensor embeddings."
    )
    segment_columns = [f"segment_{i}_cosine" for i in range(segments)]
    segment_text = ", ".join(f"S{i + 1} {results[column].mean():.3f}" for i, column in enumerate(segment_columns))
    report = f"""# Temporal HuBERT Teacher-Student Results

The audio teacher was silence-trimmed and mean-pooled into {segments} ordered relative-time
segments. The silent student predicts the resulting `{segments} x 768` HuBERT signature
from lip, UWB, mmWave, and laser embeddings; audio remains absent at inference.

## Design

- Evaluation uses the existing five speaker-disjoint, encoder-disjoint folds.
- Teacher centering is fitted separately on each training fold.
- True segment order is compared with reversed and one-step-shifted controls.
- Silent inputs remain one fixed embedding per modality, so this tests recovery of an
  ordered utterance signature rather than frame-to-frame sensor alignment.

## Per-Fold Results

| Fold | Temporal Student Accuracy | Pooled Student Accuracy | True-Order Segment Cosine | Reversed Cosine | Order Margin |
|---:|---:|---:|---:|---:|---:|
{rows}

## Aggregate

- Temporal-student class accuracy: **{100*results.accuracy.mean():.1f}% +/- {100*results.accuracy.std(ddof=1):.1f}%**.
- Pooled-HuBERT student accuracy: **{100*pooled.test_accuracy.mean():.1f}% +/- {100*pooled.test_accuracy.std(ddof=1):.1f}%**.
- True-order segment cosine: **{results.segment_cosine.mean():.3f}**.
- Reversed-order cosine: **{results.reversed_segment_cosine.mean():.3f}**.
- Shifted-order cosine: **{results.shifted_segment_cosine.mean():.3f}**.
- Mean true-versus-reversed order margin: **{margin:+.3f}**.
- Train-mean target baseline cosine: **{results.mean_target_baseline_cosine.mean():.3f}**.
- Segment-position cosines: {segment_text}.

{conclusion}

## Boundary

Relative-time targets are a stricter teacher than one global mean, but they do not create
framewise silent-sensor observations. Phoneme-level claims require either temporal sensor
encoder activations or explicit forced-alignment/articulatory labels.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--teacher-targets", default="artifacts/teacher_targets/facebook_hubert-base-ls960_temporal4_targets.npz")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="artifacts/students/hubert_temporal4_cv")
    parser.add_argument("--output", default="reports/results/hubert_temporal_teacher_student_cv.csv")
    parser.add_argument("--report-output", default="reports/hubert_temporal_teacher_student.md")
    parser.add_argument("--pooled-results", default="reports/results/hubert_teacher_student_cv_results.csv")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    started = time.perf_counter()
    config = load_config(args.config)
    teacher = load_teacher_targets(args.teacher_targets)
    if len(teacher["target_shape"]) != 2:
        raise ValueError("Temporal teacher targets must include a two-dimensional target_shape")
    folds_requested = [int(value) for value in args.folds.split(",") if value.strip()]
    seed_paths, _ = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    folds = [fold for fold in folds if int(fold["fold"]) in folds_requested]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold"])
        checkpoint_path = output_dir / f"fold_{fold_id}_teacher_student.pt"
        if not args.summarize_only:
            subprocess.run(
                [
                    sys.executable, "scripts/18_train_teacher_student.py",
                    "--config", args.config,
                    "--teacher-targets", args.teacher_targets,
                    "--fold", str(fold_id),
                    "--max-epochs", str(args.max_epochs),
                    "--batch-size", str(args.batch_size),
                    "--device", args.device,
                    "--output-dir", str(output_dir),
                ],
                check=True,
            )
        rows.append(_evaluate_fold(config, teacher, fold, checkpoint_path))
        elapsed = time.perf_counter() - started
        remaining = elapsed / position * (len(folds) - position)
        print(
            f"TEMPORAL_CV fold={fold_id} position={position}/{len(folds)} "
            f"segment_cosine={rows[-1]['segment_cosine']:.3f} estimated_remaining_seconds={remaining:.1f}",
            flush=True,
        )

    results = pd.DataFrame(rows).sort_values("fold")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    pooled = pd.read_csv(args.pooled_results)
    _write_report(Path(args.report_output), results, pooled, int(teacher["target_shape"][0]))
    print(f"Saved temporal CV report to {args.report_output}")


if __name__ == "__main__":
    main()
