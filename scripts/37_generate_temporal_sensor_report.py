#!/usr/bin/env python3
"""Generate figures and a consolidated temporal-sensor interpretability report."""

from __future__ import annotations

import html
from pathlib import Path

import pandas as pd


def _grouped_accuracy_figure(path: Path, sensor: pd.DataFrame, fixed: pd.DataFrame) -> None:
    width, height = 900, 350
    left, right, top, bottom = 90, 30, 60, 55
    chart_w, chart_h = width - left - right, height - top - bottom
    fixed = fixed.set_index("fold")
    items = ['<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">Temporal HuBERT alignment from silent inputs</text>']
    maximum = max(sensor.segment_cosine.max(), fixed.segment_cosine.max()) * 1.2
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = top + chart_h * (1 - tick)
        items.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        items.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11">{maximum*tick:.2f}</text>')
    group_w = chart_w / len(sensor)
    for position, row in enumerate(sensor.itertuples(index=False)):
        center = left + group_w * (position + 0.5)
        values = [(float(fixed.loc[int(row.fold), "segment_cosine"]), "#64748b"), (float(row.segment_cosine), "#0f766e")]
        for offset, (value, color) in zip((-30, 4), values, strict=True):
            bar_height = chart_h * value / maximum
            items.append(f'<rect x="{center+offset:.1f}" y="{top+chart_h-bar_height:.1f}" width="26" height="{bar_height:.1f}" fill="{color}" rx="2"/>')
        items.append(f'<text x="{center:.1f}" y="{height-25}" text-anchor="middle" font-family="Arial" font-size="12">Fold {int(row.fold)}</text>')
    items.extend([
        '<rect x="640" y="18" width="12" height="12" fill="#64748b"/><text x="658" y="29" font-family="Arial" font-size="11">Fixed embeddings</text>',
        '<rect x="755" y="18" width="12" height="12" fill="#0f766e"/><text x="773" y="29" font-family="Arial" font-size="11">Temporal states</text>',
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/>\n'
        + "\n".join(items) + "\n</svg>\n",
        encoding="utf-8",
    )


def _signed_bar_figure(path: Path, summary: pd.DataFrame) -> None:
    selected = summary[
        summary["representation"].isin(
            [
                "lip",
                "contactless_nonlip",
                "all_modalities",
                "temporal_student",
                "multitask_temporal_student",
                "attention_temporal_student",
            ]
        )
    ]
    rows = [(f"{row.representation.replace('_', ' ').title()} / {row.target.replace('_', ' ')}", float(row.delta_r2_mean)) for row in selected.itertuples(index=False)]
    width, left, right, top, row_h = 900, 330, 80, 62, 31
    height = top + len(rows) * row_h + 36
    chart_w = width - left - right
    max_abs = max(max(abs(value) for _, value in rows) * 1.15, 0.01)
    center = left + chart_w / 2
    items = [f'<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">{html.escape("Articulation R2 gain over class-and-position baseline")}</text>']
    items.append(f'<line x1="{center:.1f}" y1="48" x2="{center:.1f}" y2="{height-28}" stroke="#9ca3af"/>')
    for index, (label, value) in enumerate(rows):
        y = top + index * row_h
        length = chart_w / 2 * abs(value) / max_abs
        x = center if value >= 0 else center - length
        color = "#0f766e" if value >= 0 else "#b45309"
        items.append(f'<text x="24" y="{y+16}" font-family="Arial" font-size="11">{html.escape(label)}</text>')
        items.append(f'<rect x="{x:.1f}" y="{y}" width="{length:.1f}" height="20" fill="{color}" rx="2"/>')
        anchor = "start" if value >= 0 else "end"
        tx = center + length + 5 if value >= 0 else center - length - 5
        items.append(f'<text x="{tx:.1f}" y="{y+15}" text-anchor="{anchor}" font-family="Arial" font-size="10">{value:+.3f}</text>')
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="#fff"/>\n'
        + "\n".join(items) + "\n</svg>\n",
        encoding="utf-8",
    )


def main() -> None:
    results = Path("reports/results")
    sensor = pd.read_csv(results / "temporal_sensor_student_cv.csv")
    multitask_path = results / "temporal_sensor_multitask_cv.csv"
    multitask = pd.read_csv(multitask_path) if multitask_path.exists() else None
    attention_path = results / "temporal_sensor_attention_cv.csv"
    attention = pd.read_csv(attention_path) if attention_path.exists() else None
    fixed = pd.read_csv(results / "hubert_temporal_teacher_student_cv.csv")
    articulation = pd.read_csv(results / "temporal_articulation_probe_summary.csv")
    audit = pd.read_csv(results / "temporal_sensor_activation_audit.csv")
    figures = Path("reports/figures")
    _grouped_accuracy_figure(figures / "temporal_sensor_hubert_alignment.svg", sensor, fixed)
    _signed_bar_figure(figures / "temporal_sensor_articulation_gain.svg", articulation)

    articulation_rows = "\n".join(
        f"| {row.representation.replace('_', ' ').title()} | {row.target.replace('_', ' ').title()} | "
        f"{row.r2_mean:.3f} | {row.class_position_baseline_r2_mean:.3f} | {row.delta_r2_mean:+.3f} | "
        f"{row.correlation_mean:.3f} | {row.order_margin_mean:+.3f} |"
        for row in articulation.sort_values(["representation", "target"]).itertuples(index=False)
    )
    contactless = articulation[articulation["representation"] == "contactless_nonlip"]
    contactless_gain = contactless.delta_r2_mean.mean()
    contactless_motion_gain = float(contactless.loc[contactless["target"] == "lip_motion", "delta_r2_mean"].iloc[0])
    contactless_statement = (
        "The non-lip contactless sensors add held-out articulatory information beyond the class-and-position template."
        if contactless_gain > 0
        else "The non-lip contactless sensors do not improve average held-out articulation R2 beyond the class-and-position template in this experiment."
    )
    coverage_rows = "\n".join(
        f"| {row.modality.title()} | {int(row.pairs_min)}-{int(row.pairs_max)} | {row.repetitions_mean:.1f} | {row.encoder_steps_mean:.1f} |"
        for row in audit.groupby("modality", as_index=False).agg(
            pairs_min=("pairs", "min"),
            pairs_max=("pairs", "max"),
            repetitions_mean=("raw_repetitions", "mean"),
            encoder_steps_mean=("mean_encoder_steps", "mean"),
        ).itertuples(index=False)
    )
    multitask_summary = ""
    if multitask is not None:
        multitask_summary = f"""
- Multitask temporal-sensor class accuracy: **{100*multitask.accuracy.mean():.1f}% +/- {100*multitask.accuracy.std(ddof=1):.1f}%**.
- Multitask temporal-sensor true-order cosine: **{multitask.segment_cosine.mean():.3f}**.
- Multitask true-versus-reversed margin: **{multitask.order_margin_reversed.mean():+.3f}**.
- Detailed multitask comparison: [multitask report](temporal_sensor_multitask.md).
"""
    attention_summary = ""
    if attention is not None:
        attention_summary = f"""
- Modality-attention class accuracy: **{100*attention.accuracy.mean():.1f}%**.
- Modality-attention true-order cosine: **{attention.segment_cosine.mean():.3f}**.
- This branch underperforms the multitask student and remains diagnostic; see
  [attention results](temporal_sensor_attention.md) and
  [held-out weight audit](temporal_sensor_attention_audit.md).
"""
    report = f"""# Temporal Silent-Sensor Interpretability

This batch exposes sequence activations from the trained lip, laser, mmWave, and UWB
encoders and tests them against both temporal HuBERT and measured lip articulation.

![HuBERT alignment](figures/temporal_sensor_hubert_alignment.svg)

![Articulation gains](figures/temporal_sensor_articulation_gain.svg)

## Data Boundary

The local RVTALL release contains anonymous corpus identifiers (`word1`, `sentences3`,
and similar) but no transcripts, TextGrids, or phoneme timestamps. Therefore this report
does **not** claim forced phoneme alignment. Its articulatory targets are directly measured
from normalized lip landmarks: inner-lip aperture, mouth width, and lip motion.

## Temporal Activation Audit

| Modality | Pairs Per Fold | Mean Raw Repetitions | Mean Encoder Steps |
|---|---:|---:|---:|
{coverage_rows}

Each repetition is encoded by the fold-specific model, pooled into four relative-time
regions, then averaged within each speaker/utterance pair.

## Temporal HuBERT Alignment

- Temporal-sensor class accuracy: **{100*sensor.accuracy.mean():.1f}% +/- {100*sensor.accuracy.std(ddof=1):.1f}%**.
- Temporal-sensor true-order cosine: **{sensor.segment_cosine.mean():.3f}**.
- Fixed-embedding temporal-student cosine: **{fixed.segment_cosine.mean():.3f}**.
- Reversed-order temporal-sensor cosine: **{sensor.reversed_segment_cosine.mean():.3f}**.
- True-versus-reversed margin: **{sensor.order_margin_reversed.mean():+.3f}**.
{multitask_summary}
{attention_summary}

## Articulation Probes

All probes are speaker-disjoint. They predict residual articulation beyond a training-fold
class-and-segment-position template; `Delta R2` is the gain over that stronger baseline.

| Representation | Target | R2 | Class+Position Baseline R2 | Delta R2 | Correlation | Order Margin |
|---|---|---:|---:|---:|---:|---:|
{articulation_rows}

{contactless_statement} Their mean delta across aperture, width, and motion is
**{contactless_gain:+.3f} R2**, but lip motion alone improves by
**{contactless_motion_gain:+.3f} R2**.

## Interpretation Boundary

Relative-time pooling establishes ordered representational evidence, not frame-exact
synchronization. A true phoneme study still requires the original prompt text plus forced
alignment, or externally supplied phonetic/articulatory annotations.
"""
    Path("reports/temporal_sensor_interpretability.md").write_text(report, encoding="utf-8")
    print("Saved temporal sensor interpretability report")


if __name__ == "__main__":
    main()
