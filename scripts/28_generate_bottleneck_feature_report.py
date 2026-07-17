#!/usr/bin/env python3
"""Generate SAE and causal-feature reports and figures."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _bar_chart(path: Path, title: str, rows: list[tuple[str, float, str]], maximum: float = 1.0) -> None:
    width, left, right, top, row_h = 900, 250, 120, 64, 36
    height = top + len(rows) * row_h + 44
    chart_w = width - left - right
    items = [f'<text x="24" y="36" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#111827">{html.escape(title)}</text>']
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + chart_w * tick
        items.append(f'<line x1="{x:.1f}" y1="48" x2="{x:.1f}" y2="{height-34}" stroke="#e5e7eb"/>')
        items.append(f'<text x="{x:.1f}" y="{height-12}" text-anchor="middle" font-family="Arial" font-size="11">{maximum*tick:.2f}</text>')
    for index, (label, value, color) in enumerate(rows):
        y = top + index * row_h
        bar_w = chart_w * max(0.0, value) / maximum
        items.append(f'<text x="24" y="{y+17}" font-family="Arial" font-size="12">{html.escape(label)}</text>')
        items.append(f'<rect x="{left}" y="{y}" width="{bar_w:.1f}" height="21" rx="3" fill="{color}"/>')
        items.append(f'<text x="{left+bar_w+7:.1f}" y="{y+16}" font-family="Arial" font-size="11">{value:.3f}</text>')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>\n' + "\n".join(items) + "\n</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    results_dir = Path("reports/results")
    sae = pd.read_csv(results_dir / "hubert_bottleneck_sae_results.csv")
    rankings = pd.read_csv(results_dir / "hubert_bottleneck_feature_rankings.csv")
    ablations = pd.read_csv(results_dir / "hubert_bottleneck_causal_ablation.csv")
    figures_dir = Path("reports/figures")

    _bar_chart(
        figures_dir / "hubert_sae_explained_variance.svg",
        "Held-out SAE explained variance",
        [(f"Fold {int(row.fold)}", float(row.test_explained_variance), "#0f766e") for row in sae.itertuples(index=False)],
    )

    zero = ablations[(ablations["mode"] == "zero") & ablations["selection"].isin(["top", "random"])]
    cosine = zero.groupby(["selection", "k"], as_index=False)["delta_target_cosine"].mean()
    cosine_rows = []
    for k in (1, 5, 10, 20, 50):
        for selection, color in (("top", "#b45309"), ("random", "#6b7280")):
            value = -float(cosine[(cosine["selection"] == selection) & (cosine["k"] == k)]["delta_target_cosine"].iloc[0])
            cosine_rows.append((f"{selection.title()} {k}", value, color))
    _bar_chart(
        figures_dir / "hubert_feature_ablation_cosine.svg",
        "Residual-HuBERT cosine loss after zero ablation",
        cosine_rows,
        maximum=max(value for _, value, _ in cosine_rows) * 1.15,
    )

    reconstruction = ablations[ablations["selection"] == "reconstruction"]
    original = ablations[ablations["selection"] == "original"]
    zero50 = zero[zero["k"] == 50].groupby("selection").mean(numeric_only=True)
    top20 = rankings[rankings["rank"] < 20]
    top_features = rankings[rankings["rank"] < 3].sort_values(["fold", "rank"])

    sae_rows = "\n".join(
        f"| {int(row.fold)} | {100*row.test_explained_variance:.1f}% | {row.test_mse:.3f} | "
        f"{row.test_mean_active_features:.1f} | {100*row.test_dead_feature_fraction:.1f}% |"
        for row in sae.itertuples(index=False)
    )
    feature_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.feature)} | {int(row.best_class)} | {int(row.best_type)} | "
        f"{row.class_selectivity:.3f} | {row.type_selectivity:.3f} | {row.speaker_selectivity:.3f} | "
        f"{100*row.activation_frequency:.1f}% | {row.decoder_stability:.3f} |"
        for row in top_features.itertuples(index=False)
    )

    report = f"""# Bottleneck Sparse Features And Causal Ablation

Fold-specific Top-K sparse autoencoders were trained on the 64-dimensional HuBERT
student bottleneck using training speakers only. Each SAE has 512 features and exactly
32 active features per sample.

![SAE explained variance](figures/hubert_sae_explained_variance.svg)

![Feature ablation](figures/hubert_feature_ablation_cosine.svg)

## SAE Quality

| Fold | Test Explained Variance | Test MSE | Active Features | Dead Features |
|---:|---:|---:|---:|---:|
{sae_rows}

Mean held-out explained variance is **{100*sae.test_explained_variance.mean():.1f}%**.
Reconstruction changes student class accuracy from **{100*original.accuracy.mean():.1f}%**
to **{100*reconstruction.accuracy.mean():.1f}%**, so causal effects are measured relative
to reconstructed bottlenecks.

## Highest-Ranked Fold-Local Features

| Fold | Feature | Best Class | Best Type | Class Selectivity | Type Selectivity | Speaker Selectivity | Frequency | Stability |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{feature_rows}

The top-20 features have mean cross-fold decoder stability **{top20.decoder_stability.mean():.3f}**.
Feature IDs are therefore treated as fold-local rather than as globally identical concepts.

## Causal Result

Zeroing the top 50 content-ranked features, relative to SAE reconstruction:

- changes residual-HuBERT cosine by **{zero50.loc['top', 'delta_target_cosine']:.3f}**, versus
  **{zero50.loc['random', 'delta_target_cosine']:.3f}** for random features;
- changes utterance-type accuracy by **{100*zero50.loc['top', 'delta_type_accuracy']:+.1f} points**,
  versus **{100*zero50.loc['random', 'delta_type_accuracy']:+.1f} points** for random features;
- changes 30-class accuracy by **{100*zero50.loc['top', 'delta_accuracy']:+.1f} points**,
  versus **{100*zero50.loc['random', 'delta_accuracy']:+.1f} points** for random features.

The ranked features have a specific causal effect on HuBERT alignment and coarse
utterance type beyond random ablations. A larger 30-class effect also appears when 50
features are removed, but it is not monotonic at smaller intervention sizes, suggesting
that fine-grained class decisions remain more distributed or redundant.

## Boundary

This is evidence for causal contribution under feature ablation, not proof that a sparse
feature corresponds to a human-named phoneme or articulator. Temporal teachers and
sample-level feature inspection are required before assigning linguistic labels.
"""
    Path("reports/hubert_bottleneck_feature_causality.md").write_text(report, encoding="utf-8")
    print("Saved sparse-feature figures and causal report")


if __name__ == "__main__":
    main()
