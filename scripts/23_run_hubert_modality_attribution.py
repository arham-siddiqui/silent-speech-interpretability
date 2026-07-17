#!/usr/bin/env python3
"""Train single-modality and leave-one-out HuBERT students across folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


FULL_MODALITIES = ["lip", "uwb", "mmwave", "laser"]
VARIANTS = {
    "single_lip": ["lip"],
    "single_uwb": ["uwb"],
    "single_mmwave": ["mmwave"],
    "single_laser": ["laser"],
    "leave_out_lip": ["uwb", "mmwave", "laser"],
    "leave_out_uwb": ["lip", "mmwave", "laser"],
    "leave_out_mmwave": ["lip", "uwb", "laser"],
    "leave_out_laser": ["lip", "uwb", "mmwave"],
}


def _metric_row(variant: str, modalities: list[str], metrics: dict) -> dict[str, object]:
    return {
        "variant": variant,
        "fold": int(metrics["fold"]),
        "modalities": ",".join(modalities),
        "num_train": int(metrics["num_train"]),
        "num_val": int(metrics["num_val"]),
        "num_test": int(metrics["num_test"]),
        "val_accuracy": float(metrics["val"]["accuracy"]),
        "test_accuracy": float(metrics["test"]["accuracy"]),
        "test_mse": float(metrics["test"]["mse"]),
        "test_cosine_similarity": float(metrics["test"]["cosine_similarity"]),
        "mean_baseline_mse": float(metrics["target_mean_baseline"]["mse"]),
        "mean_baseline_cosine_similarity": float(metrics["target_mean_baseline"]["cosine_similarity"]),
    }


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    full_accuracy = float(summary.loc[summary["variant"] == "full", "accuracy_mean"].iloc[0])
    singles = summary[summary["variant"].str.startswith("single_")].sort_values("accuracy_mean", ascending=False)
    leave_out = summary[summary["variant"].str.startswith("leave_out_")].copy()
    best_single = singles.iloc[0]
    largest_drop = leave_out.sort_values("delta_vs_full").iloc[0]
    rows = []
    for item in summary.sort_values(["group_order", "accuracy_mean"], ascending=[True, False]).itertuples(index=False):
        rows.append(
            f"| {item.variant.replace('_', ' ').title()} | {item.modalities.replace(',', ', ')} | "
            f"{100 * item.accuracy_mean:.1f}% | {100 * item.accuracy_std:.1f}% | "
            f"{100 * item.delta_vs_full:+.1f} | {item.cosine_mean:.3f} | {item.mse_mean:.3f} |"
        )
    removed = largest_drop.variant.removeprefix("leave_out_")
    text = f"""# HuBERT Student Modality Attribution

Each student variant was trained across all five encoder-disjoint folds against the
same HuBERT targets. Every variant uses the full four-modality pair intersection, so
sample coverage and held-out speakers are identical across comparisons.

## Results

| Variant | Included Modalities | Mean Accuracy | Std. Dev. | Delta vs Full (pp) | Target Cosine | Target MSE |
|---|---|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## Main Findings

- The full four-sensor student reaches **{100 * full_accuracy:.1f}%** mean accuracy.
- The strongest single-sensor student is **{best_single.variant.removeprefix('single_')}**
  at **{100 * best_single.accuracy_mean:.1f}%**.
- Removing **{removed}** produces the largest mean accuracy drop relative to the full
  student (**{100 * largest_drop.delta_vs_full:+.1f} percentage points**).

Single-modality performance measures sufficiency, while leave-one-out changes measure
conditional contribution given the other sensors. Neither establishes causal feature
mechanisms inside the network; that requires activation-level ablation.

The train-mean HuBERT direction has mean cosine similarity
**{summary['mean_baseline_cosine_mean'].mean():.3f}** on held-out speakers. Student target
cosine should be interpreted relative to this baseline; class accuracy and target
alignment need not move together because the training objective contains both losses.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--teacher-targets", default="artifacts/teacher_targets/facebook_hubert-base-ls960_targets.npz")
    parser.add_argument("--full-student-dir", default="artifacts/students/hubert_cv")
    parser.add_argument("--output-dir", default="artifacts/students/hubert_attribution")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--results-output", default="reports/results/hubert_modality_attribution_results.csv")
    parser.add_argument("--summary-output", default="reports/results/hubert_modality_attribution_summary.csv")
    parser.add_argument("--report-output", default="reports/hubert_modality_attribution.md")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    rows = []
    for fold in range(5):
        metrics = json.loads(
            (Path(args.full_student_dir) / f"fold_{fold}_teacher_student_metrics.json").read_text(encoding="utf-8")
        )
        rows.append(_metric_row("full", FULL_MODALITIES, metrics))

    jobs = [(variant, modalities, fold) for variant, modalities in VARIANTS.items() for fold in range(5)]
    start_all = time.perf_counter()
    for position, (variant, modalities, fold) in enumerate(jobs, start=1):
        variant_dir = Path(args.output_dir) / variant
        metrics_path = variant_dir / f"fold_{fold}_teacher_student_metrics.json"
        if args.force or not metrics_path.exists():
            command = [
                sys.executable,
                "scripts/18_train_teacher_student.py",
                "--config", args.config,
                "--teacher-targets", args.teacher_targets,
                "--fold", str(fold),
                "--modalities", ",".join(modalities),
                "--pair-modalities", ",".join(FULL_MODALITIES),
                "--max-epochs", str(args.max_epochs),
                "--batch-size", str(args.batch_size),
                "--device", args.device,
                "--output-dir", str(variant_dir),
            ]
            completed = subprocess.run(command, text=True, capture_output=True)
            if completed.returncode:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
                completed.check_returncode()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(_metric_row(variant, modalities, metrics))
        elapsed = time.perf_counter() - start_all
        remaining = elapsed / position * (len(jobs) - position)
        print(
            f"ATTRIBUTION_PROGRESS variant={variant} fold={fold} position={position}/{len(jobs)} "
            f"test_accuracy={metrics['test']['accuracy']:.4f} estimated_remaining_seconds={remaining:.1f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["variant", "modalities"], as_index=False)
        .agg(
            accuracy_mean=("test_accuracy", "mean"),
            accuracy_std=("test_accuracy", "std"),
            mse_mean=("test_mse", "mean"),
            mse_std=("test_mse", "std"),
            cosine_mean=("test_cosine_similarity", "mean"),
            cosine_std=("test_cosine_similarity", "std"),
            mean_baseline_mse_mean=("mean_baseline_mse", "mean"),
            mean_baseline_cosine_mean=("mean_baseline_cosine_similarity", "mean"),
            folds=("fold", "nunique"),
        )
    )
    full_accuracy = float(summary.loc[summary["variant"] == "full", "accuracy_mean"].iloc[0])
    summary["delta_vs_full"] = summary["accuracy_mean"] - full_accuracy
    summary["group_order"] = summary["variant"].map(
        lambda value: 0 if value == "full" else 1 if value.startswith("single_") else 2
    )
    results_path = Path(args.results_output)
    summary_path = Path(args.summary_output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_report(Path(args.report_output), summary)
    print(f"Saved attribution results to {results_path}")
    print(f"Saved pushable report to {args.report_output}")


if __name__ == "__main__":
    main()
