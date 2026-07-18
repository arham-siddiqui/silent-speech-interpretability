#!/usr/bin/env python3
"""Report probes and sparse-feature causality for the temporal HuBERT student."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _figure(path: Path, pooled: pd.DataFrame, temporal: pd.DataFrame) -> None:
    width, height = 900, 350
    left, right, top, bottom = 90, 30, 65, 65
    chart_w, chart_h = width - left - right, height - top - bottom
    rows = []
    for name, frame in (("Pooled", pooled), ("Temporal", temporal)):
        zero50 = frame[(frame["mode"] == "zero") & (frame["k"] == 50)].groupby("selection").mean(numeric_only=True)
        rows.extend([
            (f"{name} top", -float(zero50.loc["top", "delta_target_cosine"]), "#b45309"),
            (f"{name} random", -float(zero50.loc["random", "delta_target_cosine"]), "#64748b"),
        ])
    maximum = max(value for _, value, _ in rows) * 1.2
    items = ['<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">Top-50 sparse-feature ablation: target cosine loss</text>']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_h * (1 - tick)
        items.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        items.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{maximum*tick:.2f}</text>')
    group_w = chart_w / len(rows)
    for position, (label, value, color) in enumerate(rows):
        x = left + group_w * position + group_w * 0.25
        bar_w = group_w * 0.5
        bar_h = chart_h * value / maximum
        items.append(f'<rect x="{x:.1f}" y="{top+chart_h-bar_h:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="{color}" rx="3"/>')
        items.append(f'<text x="{x+bar_w/2:.1f}" y="{height-35}" text-anchor="middle" font-family="Arial" font-size="12">{label}</text>')
        items.append(f'<text x="{x+bar_w/2:.1f}" y="{top+chart_h-bar_h-7:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.3f}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/>\n'
        + "\n".join(items) + "\n</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    results = Path("reports/results")
    temporal_probe = pd.read_csv(results / "hubert_temporal_student_probe_summary.csv")
    pooled_probe = pd.read_csv(results / "hubert_student_probe_summary.csv")
    temporal_sae = pd.read_csv(results / "hubert_temporal_bottleneck_sae_results.csv")
    temporal_ablation = pd.read_csv(results / "hubert_temporal_bottleneck_causal_ablation.csv")
    pooled_ablation = pd.read_csv(results / "hubert_bottleneck_causal_ablation.csv")
    temporal_zero50 = temporal_ablation[(temporal_ablation["mode"] == "zero") & (temporal_ablation["k"] == 50)].groupby("selection").mean(numeric_only=True)
    pooled_zero50 = pooled_ablation[(pooled_ablation["mode"] == "zero") & (pooled_ablation["k"] == 50)].groupby("selection").mean(numeric_only=True)
    temporal_probe = temporal_probe.set_index(["task", "layer"])
    pooled_probe = pooled_probe.set_index(["task", "layer"])
    figure = Path("reports/figures/hubert_temporal_feature_ablation.svg")
    _figure(figure, pooled_ablation, temporal_ablation)

    report = f"""# Temporal HuBERT Bottleneck Interpretability

The temporal HuBERT student was evaluated with the same frozen probes, fold-specific
Top-K sparse autoencoders, and controlled feature ablations as the pooled-HuBERT student.

![Temporal feature ablation](figures/hubert_temporal_feature_ablation.svg)

## Representation Probes

| Metric | Pooled HuBERT | Temporal HuBERT |
|---|---:|---:|
| Bottleneck class accuracy | {100*pooled_probe.loc[("utterance_class", "bottleneck"), "accuracy_mean"]:.1f}% | {100*temporal_probe.loc[("utterance_class", "bottleneck"), "accuracy_mean"]:.1f}% |
| Bottleneck type accuracy | {100*pooled_probe.loc[("utterance_type", "bottleneck"), "accuracy_mean"]:.1f}% | {100*temporal_probe.loc[("utterance_type", "bottleneck"), "accuracy_mean"]:.1f}% |
| Bottleneck speaker leakage | {100*pooled_probe.loc[("speaker_leakage", "bottleneck"), "accuracy_mean"]:.1f}% | {100*temporal_probe.loc[("speaker_leakage", "bottleneck"), "accuracy_mean"]:.1f}% |

The temporal bottleneck remains class- and type-informative while suppressing most
linearly decodable speaker identity.

## Sparse Reconstruction

The temporal bottleneck SAE explains **{100*temporal_sae.test_explained_variance.mean():.1f}%**
of held-out variance with exactly 32 of 512 features active per utterance. Reconstruction
changes class accuracy from **{100*temporal_ablation[temporal_ablation.selection == "original"].accuracy.mean():.1f}%**
to **{100*temporal_ablation[temporal_ablation.selection == "reconstruction"].accuracy.mean():.1f}%**;
ablations are measured relative to reconstruction.

## Top-50 Causal Ablation

| Effect | Pooled Top | Pooled Random | Temporal Top | Temporal Random |
|---|---:|---:|---:|---:|
| Target cosine change | {pooled_zero50.loc["top", "delta_target_cosine"]:.3f} | {pooled_zero50.loc["random", "delta_target_cosine"]:.3f} | {temporal_zero50.loc["top", "delta_target_cosine"]:.3f} | {temporal_zero50.loc["random", "delta_target_cosine"]:.3f} |
| Type accuracy change | {100*pooled_zero50.loc["top", "delta_type_accuracy"]:+.1f} pp | {100*pooled_zero50.loc["random", "delta_type_accuracy"]:+.1f} pp | {100*temporal_zero50.loc["top", "delta_type_accuracy"]:+.1f} pp | {100*temporal_zero50.loc["random", "delta_type_accuracy"]:+.1f} pp |
| Class accuracy change | {100*pooled_zero50.loc["top", "delta_accuracy"]:+.1f} pp | {100*pooled_zero50.loc["random", "delta_accuracy"]:+.1f} pp | {100*temporal_zero50.loc["top", "delta_accuracy"]:+.1f} pp | {100*temporal_zero50.loc["random", "delta_accuracy"]:+.1f} pp |

Temporal content-ranked features have a specific causal effect on teacher alignment and
coarse utterance type beyond random controls. Fine-grained 30-class decisions remain
distributed: the temporal top-50 intervention barely changes class accuracy.

## Boundary

This identifies causally important temporal-teacher features, not phoneme timestamps.
The silent inputs are still fixed utterance embeddings, so the next decisive experiment
must expose temporal sensor encoder activations or introduce forced-alignment labels.
"""
    Path("reports/hubert_temporal_feature_causality.md").write_text(report, encoding="utf-8")
    print("Saved temporal feature causality report")


if __name__ == "__main__":
    main()
