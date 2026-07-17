#!/usr/bin/env python3
"""Generate lightweight SVG figures for true encoder-disjoint CV results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


COLORS = {
    "lip": "#2f6fbb",
    "mouth": "#9aa3ad",
    "uwb": "#168a5b",
    "mmwave": "#8a5fbf",
    "laser": "#d47b24",
    "fusion": "#1f2937",
    "equal_weight": "#4b5563",
    "equal_weight_no_mouth": "#2563eb",
    "validation_weighted": "#0f766e",
    "borda": "#7c3aed",
    "consistency_weighted": "#b45309",
}

METHOD_LABELS = {
    "prototype": "Prototype",
    "equal_weight": "Equal",
    "equal_weight_no_mouth": "No-mouth equal",
    "validation_weighted": "Validation",
    "borda": "Borda",
    "consistency_weighted": "Consistency",
}

MODALITY_LABELS = {
    "lip": "Lip",
    "mouth": "Mouth",
    "uwb": "UWB",
    "mmwave": "mmWave",
    "laser": "Laser",
    "fusion": "Fusion",
}


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _svg_text(x: float, y: float, text: str, size: int = 13, weight: str = "400", anchor: str = "start") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Inter, Arial, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="#111827">{text}</text>'
    )


def _write(path: Path, body: str, width: int, height: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n{body}\n</svg>\n',
        encoding="utf-8",
    )


def _bar_chart(rows: list[tuple[str, float, str]], title: str, path: Path) -> None:
    width = 860
    left = 220
    right = 80
    top = 64
    row_h = 34
    height = top + row_h * len(rows) + 42
    chart_w = width - left - right
    max_value = max(value for _label, value, _color in rows)

    elements = [_svg_text(24, 34, title, size=20, weight="700")]
    for tick in [0.0, 0.25, 0.5, 0.75]:
        x = left + chart_w * tick / max(0.75, max_value)
        elements.append(f'<line x1="{x:.1f}" y1="{top - 18}" x2="{x:.1f}" y2="{height - 36}" stroke="#e5e7eb"/>')
        elements.append(_svg_text(x, height - 14, _pct(tick), size=11, anchor="middle"))

    for i, (label, value, color) in enumerate(rows):
        y = top + i * row_h
        bar_w = chart_w * value / max(0.75, max_value)
        elements.append(_svg_text(24, y + 19, label, size=13))
        elements.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="20" rx="3" fill="{color}"/>')
        elements.append(_svg_text(left + bar_w + 8, y + 16, _pct(value), size=12))

    _write(path, "\n".join(elements), width, height)


def _weights_heatmap(weights: pd.DataFrame, path: Path) -> None:
    modalities = ["lip", "mouth", "uwb", "mmwave", "laser"]
    width = 760
    height = 320
    left = 110
    top = 72
    cell_w = 112
    cell_h = 38
    elements = [_svg_text(24, 34, "Validation-derived modality weights", size=20, weight="700")]
    for j, modality in enumerate(modalities):
        elements.append(_svg_text(left + j * cell_w + cell_w / 2, top - 18, MODALITY_LABELS[modality], size=12, anchor="middle"))
    for i, fold in enumerate(sorted(weights["fold"].unique())):
        elements.append(_svg_text(28, top + i * cell_h + 24, f"Fold {int(fold)}", size=12))
        fold_rows = weights[weights["fold"] == fold].set_index("modality")
        for j, modality in enumerate(modalities):
            value = float(fold_rows.loc[modality, "weight"])
            intensity = int(245 - 150 * value / max(weights["weight"].max(), 1e-8))
            fill = f"rgb({intensity},{235},{225})"
            x = left + j * cell_w
            y = top + i * cell_h
            elements.append(f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" rx="3" fill="{fill}" stroke="#ffffff"/>')
            elements.append(_svg_text(x + cell_w / 2 - 2, y + 23, _pct(value), size=12, anchor="middle"))
    _write(path, "\n".join(elements), width, height)


def _per_class_chart(per_class: pd.DataFrame, path: Path) -> None:
    subset = per_class[(per_class["method"] == "validation_weighted") & (per_class["modality"] == "fusion")]
    rows = [
        (f"Class {int(item.class_id)}", float(item.accuracy), "#0f766e")
        for item in subset.sort_values("class_id").itertuples(index=False)
    ]
    _bar_chart(rows, "Validation-weighted fusion accuracy by class", path)


def _method_summary_chart(summary: pd.DataFrame, path: Path) -> None:
    ordered = summary.sort_values("mean", ascending=True)
    rows = []
    for item in ordered.itertuples(index=False):
        label = f"{METHOD_LABELS.get(item.method, item.method)} / {MODALITY_LABELS.get(item.modality, item.modality)}"
        color = COLORS.get(item.method, COLORS.get(item.modality, "#4b5563"))
        rows.append((label, float(item.mean), color))
    _bar_chart(rows, "True encoder-disjoint CV mean accuracy", path)


def _write_error_report(per_class: pd.DataFrame, output_path: Path) -> None:
    subset = per_class[(per_class["method"] == "validation_weighted") & (per_class["modality"] == "fusion")].copy()
    hardest = subset.sort_values(["accuracy", "num_samples"], ascending=[True, False]).head(10)
    strongest = subset.sort_values(["accuracy", "num_samples"], ascending=[False, False]).head(10)

    def table(df: pd.DataFrame) -> str:
        rows = ["| Class ID | Accuracy | Correct | Samples |", "|---:|---:|---:|---:|"]
        for item in df.itertuples(index=False):
            rows.append(f"| {int(item.class_id)} | {_pct(item.accuracy)} | {int(item.num_correct)} | {int(item.num_samples)} |")
        return "\n".join(rows)

    output_path.write_text(
        f"""# True CV Error Analysis

This report focuses on `validation_weighted` fusion, the current strict-CV baseline.

![Per-class accuracy](figures/true_cv_per_class_accuracy.svg)

## Hardest Classes

{table(hardest)}

## Strongest Classes

{table(strongest)}

## How To Use This

The hardest classes are the first targets for confusion-matrix review and modality
ablation. If a class is weak under validation-weighted fusion but strong under one
individual modality, that class is a candidate for class-conditional or reliability-aware
fusion improvements.
""",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="reports/results")
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--error-report", default="reports/true_cv_error_analysis.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    figures_dir = Path(args.figures_dir)
    summary = pd.read_csv(results_dir / "true_encoder_cv_summary.csv")
    weights = pd.read_csv(results_dir / "true_encoder_cv_fusion_weights.csv")
    per_class = pd.read_csv(results_dir / "true_encoder_cv_per_class.csv")

    _method_summary_chart(summary, figures_dir / "true_cv_mean_accuracy.svg")
    _weights_heatmap(weights, figures_dir / "true_cv_fusion_weights.svg")
    _per_class_chart(per_class, figures_dir / "true_cv_per_class_accuracy.svg")
    _write_error_report(per_class, Path(args.error_report))


if __name__ == "__main__":
    main()
