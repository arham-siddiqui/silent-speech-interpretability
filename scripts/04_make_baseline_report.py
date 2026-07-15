#!/usr/bin/env python3
"""Generate a compact baseline report from audit, fixed split, and CV outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.evals.metrics import confusion_matrix


def _plot_outputs(results_dir: Path, figures_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"Skipping report plots because matplotlib is unavailable: {exc}")
        return

    figures_dir.mkdir(parents=True, exist_ok=True)
    per_modality = pd.read_csv(results_dir / "per_modality_results.csv")
    fixed = pd.read_csv(results_dir / "fixed_split_results.csv")
    combined = pd.concat([per_modality, fixed], ignore_index=True)
    combined["label"] = combined.apply(
        lambda row: row["modality"] if row["method"] == "prototype" else row["method"],
        axis=1,
    )
    combined = combined.sort_values("accuracy", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(combined["label"], combined["accuracy"])
    plt.ylabel("Fixed Split Accuracy")
    plt.ylim(0, 1)
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(figures_dir / "fixed_split_accuracy_bar.png", dpi=180)
    plt.close()

    predictions = pd.read_csv(results_dir / "fixed_split_predictions.csv")
    fusion = fixed.sort_values("accuracy", ascending=False).iloc[0]
    selected = predictions[(predictions["method"] == fusion["method"]) & (predictions["modality"] == "fusion")]
    labels = np.arange(max(int(selected["y_true"].max()), int(selected["y_pred"].max())) + 1)
    matrix = confusion_matrix(selected["y_true"].to_numpy(), selected["y_pred"].to_numpy(), labels)

    plt.figure(figsize=(8, 7))
    plt.imshow(matrix, cmap="Blues")
    plt.title(f"Fixed Split Confusion Matrix: {fusion['method']}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix_fixed_split.png", dpi=180)
    plt.close()


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    shown = df[columns].copy()
    for column in shown.select_dtypes(include=["float"]).columns:
        shown[column] = shown[column].map(lambda value: f"{value:.3f}")
    shown = shown.fillna("")
    rows = [[str(value) for value in row] for row in shown.to_numpy().tolist()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    results_dir = Path(config["data"]["results_dir"])
    figures_dir = Path(config["data"]["figures_dir"])
    reports_dir = Path("reports")

    audit = json.loads((results_dir / "dataset_audit.json").read_text(encoding="utf-8"))
    per_modality = pd.read_csv(results_dir / "per_modality_results.csv")
    fixed = pd.read_csv(results_dir / "fixed_split_results.csv")
    cv_summary = pd.read_csv(results_dir / "speaker_cv_summary.csv")
    comparison_path = results_dir / "baseline_comparison.csv"
    comparison = pd.read_csv(comparison_path) if comparison_path.exists() else pd.DataFrame()
    sanity_path = results_dir / "evaluation_sanity_audit.json"
    sanity = json.loads(sanity_path.read_text(encoding="utf-8")) if sanity_path.exists() else {}

    _plot_outputs(results_dir, figures_dir)

    alignment = audit.get("alignment", {})
    sources = {
        modality: info.get("source", "unknown")
        for modality, info in alignment.get("embedding_paths", {}).items()
        if info.get("exists")
    }
    best_fixed = fixed.sort_values("accuracy", ascending=False).iloc[0]
    best_cv = cv_summary.sort_values("mean", ascending=False).iloc[0]

    report = [
        "# Silent Speech Interpretability Baseline Report",
        "",
        "This report summarizes the current contactless / microphone-free speech decoding baseline.",
        "",
        "## Dataset Audit",
        "",
        f"- Manifest samples: {audit['num_samples']}",
        f"- Strict five-modality intersection: {audit['strict_intersection_samples']}",
        f"- Embedding strict intersection: {alignment.get('strict_intersection_group_count', 'n/a')}",
        f"- Label mismatches: {alignment.get('label_mismatch_count', 'n/a')}",
        f"- User ID mismatches: {alignment.get('user_id_mismatch_count', 'n/a')}",
        f"- Embedding sources: {sources}",
        "",
        "## Fixed Speaker Split",
        "",
        _markdown_table(pd.concat([per_modality, fixed], ignore_index=True), ["method", "modality", "num_train", "num_test", "accuracy", "macro_f1"]),
        "",
        f"Best fixed-split method: `{best_fixed['method']}` at accuracy {best_fixed['accuracy']:.3f}.",
        "",
        "## Legacy-Compatible Comparison",
        "",
        _markdown_table(
            comparison,
            ["baseline_family", "method", "num_train", "num_val", "num_test", "accuracy", "macro_f1"],
        )
        if not comparison.empty
        else "Run `scripts/05_compare_legacy_baseline.py` to generate this comparison.",
        "",
        "## 5-Fold Speaker-Disjoint CV",
        "",
        (
            f"Sanity note: {sanity.get('num_encoder_disjoint_cv_folds', 'n/a')}/"
            f"{sanity.get('num_cv_folds', 'n/a')} CV folds are encoder-disjoint for these precomputed embeddings. "
            "Treat this table as fusion-layer CV unless encoders are retrained inside each fold."
        )
        if sanity
        else "Sanity note: run `scripts/06_evaluation_sanity_audit.py` before interpreting CV.",
        "",
        _markdown_table(cv_summary, ["method", "modality", "mean", "std", "count"]),
        "",
        f"Best CV method: `{best_cv['method']}` / `{best_cv['modality']}` at mean accuracy {best_cv['mean']:.3f}.",
        "",
        "## Figures",
        "",
        f"- `{figures_dir / 'fixed_split_accuracy_bar.png'}`",
        f"- `{figures_dir / 'confusion_matrix_fixed_split.png'}`",
        f"- `{figures_dir / 'speaker_cv_accuracy.png'}`",
        f"- `{figures_dir / 'speaker_cv_by_modality.png'}`",
        "",
        "## Notes",
        "",
        "- Audio is not used in this baseline inference path.",
        "- Fusion metrics use the strict multimodal intersection.",
        "- Individual modality fixed-split metrics use each modality's available test pairs.",
        "- Current CV uses precomputed embeddings, so it is not full encoder-disjoint CV unless encoders are retrained per fold.",
        "",
    ]
    output = reports_dir / "baseline_report.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
