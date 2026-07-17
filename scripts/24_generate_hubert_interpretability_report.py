#!/usr/bin/env python3
"""Generate figures and a consolidated HuBERT interpretability report."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


LAYER_LABELS = {
    "sensor_input": "Sensor input",
    "hidden": "Hidden",
    "bottleneck": "Bottleneck",
    "predicted_hubert": "Predicted HuBERT",
    "teacher_hubert": "Teacher HuBERT",
}


def _write_bar_chart(path: Path, title: str, rows: list[tuple[str, float, str]]) -> None:
    width, left, right, top, row_height = 900, 260, 80, 66, 34
    height = top + len(rows) * row_height + 45
    chart_width = width - left - right
    elements = [
        f'<text x="24" y="36" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>'
    ]
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + chart_width * tick
        elements.append(f'<line x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="{height - 35}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{x:.1f}" y="{height - 14}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#374151">{100*tick:.0f}%</text>')
    for index, (label, value, color) in enumerate(rows):
        y = top + index * row_height
        elements.append(f'<text x="24" y="{y + 17}" font-family="Arial, sans-serif" font-size="12" fill="#111827">{html.escape(label)}</text>')
        elements.append(f'<rect x="{left}" y="{y}" width="{chart_width * value:.1f}" height="20" rx="3" fill="{color}"/>')
        elements.append(f'<text x="{left + chart_width * value + 7:.1f}" y="{y + 16}" font-family="Arial, sans-serif" font-size="11" fill="#111827">{100*value:.1f}%</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n' + "\n".join(elements) + "\n</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    results_dir = Path("reports/results")
    figures_dir = Path("reports/figures")
    probes = pd.read_csv(results_dir / "hubert_student_probe_summary.csv")
    attribution = pd.read_csv(results_dir / "hubert_modality_attribution_summary.csv")
    student_cv = pd.read_csv(results_dir / "hubert_teacher_student_cv_summary.csv").iloc[0]

    task_colors = {"utterance_class": "#2563eb", "utterance_type": "#0f766e", "speaker_leakage": "#b45309"}
    probe_rows = []
    for task in ("utterance_class", "utterance_type", "speaker_leakage"):
        subset = probes[probes["task"] == task].set_index("layer")
        for layer in LAYER_LABELS:
            probe_rows.append(
                (f"{task.replace('_', ' ').title()} / {LAYER_LABELS[layer]}", float(subset.loc[layer, "accuracy_mean"]), task_colors[task])
            )
    _write_bar_chart(figures_dir / "hubert_student_probe_accuracy.svg", "HuBERT student linear-probe accuracy", probe_rows)

    attribution_rows = [
        (row.variant.replace("_", " ").title(), float(row.accuracy_mean), "#0f766e" if row.variant == "full" else "#4b5563")
        for row in attribution.sort_values("accuracy_mean", ascending=False).itertuples(index=False)
    ]
    _write_bar_chart(figures_dir / "hubert_modality_attribution.svg", "HuBERT student modality attribution", attribution_rows)

    probe_index = probes.set_index(["task", "layer"])
    attr_index = attribution.set_index("variant")
    report = f"""# HuBERT Student Interpretability Summary

This report consolidates the first real audio-teacher interpretability batch for the
contactless / microphone-free student.

![Probe accuracy](figures/hubert_student_probe_accuracy.svg)

![Modality attribution](figures/hubert_modality_attribution.svg)

## Core Results

- Five-fold student accuracy: **{100 * student_cv.test_accuracy_mean:.1f}%**, compared
  with **{100 * student_cv.baseline_accuracy_mean:.1f}%** for strict validation-weighted fusion.
- Residual-HuBERT cosine: **{student_cv.test_cosine_mean:.3f}**, versus
  **{student_cv.mean_baseline_cosine:.3f}** for the train-mean residual direction.
- Bottleneck class probe: **{100 * probe_index.loc[('utterance_class', 'bottleneck'), 'accuracy_mean']:.1f}%**.
- Bottleneck utterance-type probe: **{100 * probe_index.loc[('utterance_type', 'bottleneck'), 'accuracy_mean']:.1f}%**.
- Speaker leakage falls from **{100 * probe_index.loc[('speaker_leakage', 'sensor_input'), 'accuracy_mean']:.1f}%**
  at sensor input to **{100 * probe_index.loc[('speaker_leakage', 'bottleneck'), 'accuracy_mean']:.1f}%**
  at the bottleneck.
- Lip alone reaches **{100 * attr_index.loc['single_lip', 'accuracy_mean']:.1f}%**; removing lip drops
  accuracy by **{abs(100 * attr_index.loc['leave_out_lip', 'delta_vs_full']):.1f} points**.

## Interpretation

The 64-dimensional bottleneck preserves utterance class and coarse speech-type
information while removing most linearly decodable speaker identity. Lip is largely
sufficient for class decoding, while laser provides the largest auxiliary leave-one-out
gain. UWB and mmWave have measurable standalone information but little conditional
accuracy contribution once lip and the other sensors are present.

Centering HuBERT targets with training-fold statistics was essential. Without centering,
a trivial shared mean direction achieved very high cosine similarity and obscured
utterance-varying alignment. All final results use centered targets without test-speaker
statistics.

## Limits And Next Step

Linear probes establish decodability, and modality retraining establishes attribution at
the input level; neither proves that individual bottleneck features causally control a
speech property. The next phase should train a sparse autoencoder on the bottleneck,
rank features by class/type selectivity, then ablate the top features across held-out
speakers.
"""
    Path("reports/hubert_interpretability_summary.md").write_text(report, encoding="utf-8")
    print("Saved HuBERT interpretability figures and consolidated report")


if __name__ == "__main__":
    main()
