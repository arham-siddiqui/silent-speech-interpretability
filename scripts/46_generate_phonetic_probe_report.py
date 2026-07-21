#!/usr/bin/env python3
"""Generate the tracked summary for temporal phonetic occupancy probes."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default="reports/results/temporal_phonetic_probe_summary.csv")
    parser.add_argument("--output", default="reports/temporal_phonetic_probes.md")
    parser.add_argument("--tracked-summary", default="reports/tables/temporal_phonetic_probe_summary.csv")
    args = parser.parse_args()

    summary = pd.read_csv(args.summary)
    tracked_summary = Path(args.tracked_summary)
    tracked_summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(tracked_summary, index=False)
    macro = summary.groupby("representation", as_index=False).agg(
        r2_mean=("r2_mean", "mean"), baseline_r2=("class_position_baseline_r2_mean", "mean"),
        delta_r2=("delta_r2_mean", "mean"), correlation=("correlation_mean", "mean"),
        order_margin=("order_margin_mean", "mean"),
    ).sort_values("delta_r2", ascending=False)
    best_rows = summary.loc[summary.groupby("feature").delta_r2_mean.idxmax()].sort_values("feature")
    macro_table = "\n".join(
        f"| {row.representation} | {row.r2_mean:.3f} | {row.baseline_r2:.3f} | {row.delta_r2:+.3f} | {row.correlation:.3f} | {row.order_margin:+.3f} |"
        for row in macro.itertuples(index=False)
    )
    feature_table = "\n".join(
        f"| {row.feature} | {row.representation} | {row.r2_mean:.3f} | {row.class_position_baseline_r2_mean:.3f} | {row.delta_r2_mean:+.3f} | {row.correlation_mean:.3f} |"
        for row in best_rows.itertuples(index=False)
    )
    best = macro.iloc[0]
    report = f"""# Temporal Phonetic Probes

## Result

The strongest macro-average residual probe is **{best.representation}**, with R2
**{best.r2_mean:.3f}** versus the utterance-class/position baseline **{best.baseline_r2:.3f}**
(delta **{best.delta_r2:+.3f}**). The residual design asks whether a representation explains
speaker-specific timing after the expected phonetic trajectory for each class is removed.

| Representation | R2 | Class + position baseline | Delta R2 | Correlation | Order margin |
|---|---:|---:|---:|---:|---:|
{macro_table}

## Best Representation Per Feature

| Feature | Representation | R2 | Baseline R2 | Delta R2 | Correlation |
|---|---|---:|---:|---:|---:|
{feature_table}

## Interpretation Boundary

These targets combine CTC-aligned word boundaries with uniformly interpolated ARPAbet
phones. Results support broad, time-varying phonetic **occupancy** only when they improve
over the class/position baseline and retain temporal order. They do not establish exact
phone boundaries or a one-neuron/one-phoneme correspondence. The main analysis excludes
word alignments below confidence 0.05 and retains known isolated-vowel intervals.
"""
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Saved phonetic probe report to {args.output}")


if __name__ == "__main__":
    main()
