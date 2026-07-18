#!/usr/bin/env python3
"""Generate the consolidated feature-exemplar and temporal-teacher report."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _accuracy_figure(path: Path, temporal: pd.DataFrame, pooled: pd.DataFrame) -> None:
    width, height = 900, 350
    left, right, top, bottom = 90, 30, 60, 55
    chart_w, chart_h = width - left - right, height - top - bottom
    pooled_by_fold = pooled.set_index("fold")
    items = [f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{html.escape("Pooled versus temporal HuBERT student accuracy")}</text>']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_h * (1 - tick)
        items.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        items.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{tick:.2f}</text>')
    group_w = chart_w / len(temporal)
    bar_w = 28
    for position, row in enumerate(temporal.itertuples(index=False)):
        center = left + group_w * (position + 0.5)
        values = [(float(pooled_by_fold.loc[int(row.fold), "test_accuracy"]), "#64748b"), (float(row.accuracy), "#0f766e")]
        for offset, (value, color) in zip((-bar_w, 4), values, strict=True):
            h = chart_h * value
            items.append(f'<rect x="{center+offset:.1f}" y="{top+chart_h-h:.1f}" width="{bar_w-4}" height="{h:.1f}" fill="{color}" rx="2"/>')
        items.append(f'<text x="{center:.1f}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="12">Fold {int(row.fold)}</text>')
    items.extend([
        '<rect x="650" y="18" width="12" height="12" fill="#64748b"/><text x="668" y="29" font-family="Arial" font-size="11">Pooled</text>',
        '<rect x="730" y="18" width="12" height="12" fill="#0f766e"/><text x="748" y="29" font-family="Arial" font-size="11">Temporal</text>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/>\n'
        + "\n".join(items) + "\n</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    temporal = pd.read_csv("reports/results/hubert_temporal_teacher_student_cv.csv")
    pooled = pd.read_csv("reports/results/hubert_teacher_student_cv_results.csv")
    exemplars = pd.read_csv("reports/results/hubert_sparse_feature_exemplar_summary.csv")
    temporal_ablation = pd.read_csv("reports/results/hubert_temporal_bottleneck_causal_ablation.csv")
    temporal_zero50 = temporal_ablation[(temporal_ablation["mode"] == "zero") & (temporal_ablation["k"] == 50)].groupby("selection").mean(numeric_only=True)
    figure = Path("reports/figures/hubert_pooled_vs_temporal_accuracy.svg")
    _accuracy_figure(figure, temporal, pooled)
    report = f"""# Feature Characterization And Temporal HuBERT Batch

This batch moves the audio-teacher track from utterance-level causal evidence toward
linguistic characterization.

![Pooled versus temporal student accuracy](figures/hubert_pooled_vs_temporal_accuracy.svg)

## Completed

1. The 15 highest-ranked fold-local sparse features were checked on held-out speakers.
   Mean held-out class eta-squared is **{exemplars.test_class_selectivity.mean():.3f}** and
   type eta-squared is **{exemplars.test_type_selectivity.mean():.3f}**.
2. Silence-trimmed HuBERT states were represented as four ordered relative-time segments.
3. A five-fold sensor student was trained against those temporal targets.
4. True-order alignment was compared with reversed and shifted temporal controls.
5. The temporal bottleneck was probed, sparsified, and causally ablated with random controls.

## Main Comparison

- Pooled-HuBERT student class accuracy: **{100*pooled.test_accuracy.mean():.1f}%**.
- Temporal-HuBERT student class accuracy: **{100*temporal.accuracy.mean():.1f}%**.
- Temporal true-order segment cosine: **{temporal.segment_cosine.mean():.3f}**.
- Temporal reversed-order cosine: **{temporal.reversed_segment_cosine.mean():.3f}**.
- True-versus-reversed margin: **{temporal.order_margin_reversed.mean():+.3f}**.
- Temporal top-50 feature ablation changes target cosine by
  **{temporal_zero50.loc['top', 'delta_target_cosine']:.3f}** versus
  **{temporal_zero50.loc['random', 'delta_target_cosine']:.3f}** randomly.
- Temporal top-50 feature ablation changes utterance-type accuracy by
  **{100*temporal_zero50.loc['top', 'delta_type_accuracy']:+.1f} points** versus
  **{100*temporal_zero50.loc['random', 'delta_type_accuracy']:+.1f} points** randomly.

Detailed evidence is in:

- `reports/hubert_sparse_feature_exemplars.md`
- `reports/hubert_temporal_teacher_student.md`
- `reports/hubert_temporal_feature_causality.md`
- `reports/hubert_bottleneck_feature_causality.md`
- `reports/temporal_sensor_interpretability.md`
- `reports/temporal_sensor_multitask.md`

## Grand-Scheme Status

The project now has a strict supervised baseline, a real pooled audio teacher, causal
sparse bottleneck features, held-out feature exemplars, and a first temporal-teacher
comparison. Temporal sensor states and measured lip-articulation probes are now complete.
A validation-selected multitask temporal student recovers accuracy from 49.9% to 60.1%
while retaining 0.386 ordered HuBERT cosine. The remaining interpretability gap is phoneme
naming, which needs prompt text plus forced alignment or external phonetic annotations
unavailable in the local release.
"""
    Path("reports/temporal_interpretability_batch.md").write_text(report, encoding="utf-8")
    print("Saved consolidated temporal interpretability report")


if __name__ == "__main__":
    main()
