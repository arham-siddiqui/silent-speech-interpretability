#!/usr/bin/env python3
"""Run and summarize speaker-disjoint teacher-student cross-validation."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def _result_row(metrics: dict, elapsed_seconds: float) -> dict[str, object]:
    return {
        "fold": int(metrics["fold"]),
        "seed": int(metrics["seed"]),
        "modalities": ",".join(metrics["modalities"]),
        "num_train": int(metrics["num_train"]),
        "num_val": int(metrics["num_val"]),
        "num_test": int(metrics["num_test"]),
        "val_accuracy": float(metrics["val"]["accuracy"]),
        "val_mse": float(metrics["val"]["mse"]),
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_mse": float(metrics["test"]["mse"]),
        "test_cosine_similarity": float(metrics["test"]["cosine_similarity"]),
        "mean_baseline_cosine_similarity": float(metrics["target_mean_baseline"]["cosine_similarity"]),
        "test_ce": float(metrics["test"]["ce"]),
        "test_loss": float(metrics["test"]["loss"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def _write_report(
    path: Path,
    results: pd.DataFrame,
    teacher_targets: Path,
    max_epochs: int,
) -> None:
    accuracy_mean = results["test_accuracy"].mean()
    accuracy_std = results["test_accuracy"].std(ddof=1)
    mse_mean = results["test_mse"].mean()
    mse_std = results["test_mse"].std(ddof=1)
    cosine_mean = results["test_cosine_similarity"].mean()
    cosine_std = results["test_cosine_similarity"].std(ddof=1)
    mean_baseline_cosine = results["mean_baseline_cosine_similarity"].mean()
    baseline_mean = results["baseline_accuracy"].mean()
    baseline_std = results["baseline_accuracy"].std(ddof=1)
    gap = results["accuracy_delta"].mean()
    gap_std = results["accuracy_delta"].std(ddof=1)
    fold_wins = int((results["accuracy_delta"] > 0).sum())
    rows = "\n".join(
        f"| {int(row.fold)} | {int(row.num_train)} | {int(row.num_val)} | {int(row.num_test)} | "
        f"{100 * row.val_accuracy:.1f}% | {100 * row.test_accuracy:.1f}% | "
        f"{100 * row.baseline_accuracy:.1f}% | {100 * row.accuracy_delta:+.1f} | "
        f"{row.test_cosine_similarity:.3f} | {row.test_mse:.4f} |"
        for row in results.itertuples(index=False)
    )
    comparison = "above" if gap >= 0 else "below"
    text = f"""# HuBERT Teacher-Student CV Results

This experiment trains the silent-sensor student against mean-pooled final-layer
HuBERT targets using fold-specific, encoder-disjoint sensor embeddings.

## Setup

- Teacher: `facebook/hubert-base-ls960`
- Teacher targets: `{teacher_targets}`
- Sensor modalities: lip, UWB, mmWave, and laser
- Mouth: excluded, matching the strict fusion baseline
- Evaluation: 5-fold speaker-disjoint CV
- Maximum epochs per fold: {max_epochs}, with validation-loss early stopping
- HuBERT targets are centered using training-fold statistics before normalization
- Audio is used only to create fixed teacher targets and is absent from student inference

## Per-Fold Results

| Fold | Train | Validation | Test | Validation Accuracy | Student Accuracy | Fusion Baseline | Delta (pp) | Target Cosine | Target MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Aggregate

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Student test accuracy | {100 * accuracy_mean:.1f}% | {100 * accuracy_std:.1f}% |
| Strict fusion test accuracy | {100 * baseline_mean:.1f}% | {100 * baseline_std:.1f}% |
| Paired accuracy delta | {100 * gap:+.1f} pp | {100 * gap_std:.1f} pp |
| Residual-HuBERT cosine | {cosine_mean:.3f} | {cosine_std:.3f} |
| Test target MSE | {mse_mean:.4f} | {mse_std:.4f} |

The student classifier is **{abs(100 * gap):.1f} percentage points {comparison}** the
current validation-weighted strict fusion baseline and wins on {fold_wins} of 5 folds.
The paired fold deltas vary substantially, so this small mean difference is not evidence
of a reliable improvement by itself. Target MSE separately measures how well silent
sensors recover the teacher representation.

The train-mean residual direction scores {mean_baseline_cosine:.3f} cosine on held-out
speakers, compared with {cosine_mean:.3f} for the student. Centering removes HuBERT's
dominant shared direction, so this measures recovery of utterance-varying structure.

## Interpretation Boundary

These targets are utterance-level mean-pooled HuBERT states. They establish the first
real audio-teacher baseline, but they do not yet provide frame-level articulatory or
syllable interpretation. Temporal HuBERT, SPARC, or Sylber targets remain follow-up
experiments.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument(
        "--teacher-targets",
        default="artifacts/teacher_targets/facebook_hubert-base-ls960_targets.npz",
    )
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="artifacts/students/hubert_cv")
    parser.add_argument("--results-output", default="reports/results/hubert_teacher_student_cv_results.csv")
    parser.add_argument("--summary-output", default="reports/results/hubert_teacher_student_cv_summary.csv")
    parser.add_argument("--report-output", default="reports/hubert_teacher_student_cv.md")
    parser.add_argument("--baseline-results", default="reports/results/true_encoder_cv_results.csv")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    folds = [int(value.strip()) for value in args.folds.split(",") if value.strip()]
    if not folds:
        raise ValueError("At least one fold is required.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_output)
    previous_elapsed = {}
    if results_path.exists():
        previous = pd.read_csv(results_path)
        previous_elapsed = dict(zip(previous["fold"].astype(int), previous["elapsed_seconds"].astype(float)))
    rows = []
    total_start = time.perf_counter()

    for position, fold in enumerate(folds, start=1):
        metrics_path = output_dir / f"fold_{fold}_teacher_student_metrics.json"
        if args.summarize_only:
            elapsed = previous_elapsed.get(fold, 0.0)
        else:
            print(f"CV_PROGRESS fold={fold} status=starting position={position}/{len(folds)}", flush=True)
            start = time.perf_counter()
            subprocess.run(
                [
                    sys.executable,
                    "scripts/18_train_teacher_student.py",
                    "--config",
                    args.config,
                    "--teacher-targets",
                    args.teacher_targets,
                    "--fold",
                    str(fold),
                    "--max-epochs",
                    str(args.max_epochs),
                    "--batch-size",
                    str(args.batch_size),
                    "--device",
                    args.device,
                    "--output-dir",
                    str(output_dir),
                ],
                check=True,
            )
            elapsed = time.perf_counter() - start
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(_result_row(metrics, elapsed))
        if not args.summarize_only:
            average = (time.perf_counter() - total_start) / position
            remaining = average * (len(folds) - position)
            print(
                f"CV_PROGRESS fold={fold} status=complete elapsed_seconds={elapsed:.1f} "
                f"test_accuracy={metrics['test']['accuracy']:.4f} estimated_remaining_seconds={remaining:.1f}",
                flush=True,
            )

    results = pd.DataFrame(rows).sort_values("fold")
    baseline = pd.read_csv(args.baseline_results)
    baseline = baseline[(baseline["method"] == "validation_weighted") & (baseline["modality"] == "fusion")]
    baseline_by_fold = dict(zip(baseline["fold"].astype(int), baseline["accuracy"].astype(float)))
    missing_baselines = sorted(set(results["fold"]) - set(baseline_by_fold))
    if missing_baselines:
        raise ValueError(f"Missing strict fusion baselines for folds: {missing_baselines}")
    results["baseline_accuracy"] = results["fold"].map(baseline_by_fold)
    results["accuracy_delta"] = results["test_accuracy"] - results["baseline_accuracy"]
    summary = pd.DataFrame(
        [
            {
                "folds": len(results),
                "test_accuracy_mean": results["test_accuracy"].mean(),
                "test_accuracy_std": results["test_accuracy"].std(ddof=1),
                "test_mse_mean": results["test_mse"].mean(),
                "test_mse_std": results["test_mse"].std(ddof=1),
                "test_cosine_mean": results["test_cosine_similarity"].mean(),
                "test_cosine_std": results["test_cosine_similarity"].std(ddof=1),
                "mean_baseline_cosine": results["mean_baseline_cosine_similarity"].mean(),
                "baseline_accuracy_mean": results["baseline_accuracy"].mean(),
                "baseline_accuracy_std": results["baseline_accuracy"].std(ddof=1),
                "accuracy_gap_mean": results["accuracy_delta"].mean(),
                "accuracy_gap_std": results["accuracy_delta"].std(ddof=1),
                "student_fold_wins": int((results["accuracy_delta"] > 0).sum()),
                "elapsed_seconds": results["elapsed_seconds"].sum(),
            }
        ]
    )
    summary_path = Path(args.summary_output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_report(
        Path(args.report_output),
        results,
        Path(args.teacher_targets),
        args.max_epochs,
    )
    print(f"Saved CV results to {results_path}", flush=True)
    print(f"Saved CV summary to {summary_path}", flush=True)
    print(f"Saved pushable report to {args.report_output}", flush=True)


if __name__ == "__main__":
    main()
