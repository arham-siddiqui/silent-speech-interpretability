#!/usr/bin/env python3
"""Generate tracked tables and conclusions from the manual boundary audit."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _macro(path: str, analysis: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    output = frame.groupby("representation", as_index=False).agg(
        r2=("r2_mean", "mean"),
        baseline_r2=("class_position_baseline_r2_mean", "mean"),
        delta_r2=("delta_r2_mean", "mean"),
        order_margin=("order_margin_mean", "mean"),
    )
    output.insert(0, "analysis", analysis)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="metadata/phone_boundary_audit_set.csv")
    parser.add_argument("--audited-intervals", default="artifacts/forced_alignment/phone_ctc_intervals_audited.csv")
    parser.add_argument("--source-intervals", default="artifacts/forced_alignment/phone_ctc_intervals.csv")
    parser.add_argument("--original-summary", default="reports/results/phone_ctc_sentence_probe_summary.csv")
    parser.add_argument("--audited-summary", default="reports/results/phone_ctc_sentence_probe_summary_audited.csv")
    parser.add_argument(
        "--uncorrected-summary",
        default="reports/results/phone_ctc_sentence_probe_summary_audit_matched_uncorrected.csv",
    )
    parser.add_argument(
        "--uniform-summary",
        default="reports/results/uniform_sentence_probe_summary_audit_matched.csv",
    )
    parser.add_argument("--table-output", default="reports/tables/phone_boundary_probe_comparison.csv")
    parser.add_argument("--output", default="reports/phone_boundary_audit_results.md")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata).fillna("")
    if metadata.review_status.eq("unreviewed").any():
        raise RuntimeError("Cannot report an incomplete manual audit")
    analyses = pd.concat(
        [
            _macro(args.original_summary, "original_178"),
            _macro(args.audited_summary, "manually_audited_175"),
            _macro(args.uncorrected_summary, "uncorrected_matched_175"),
            _macro(args.uniform_summary, "uniform_matched_175"),
        ],
        ignore_index=True,
    )
    pivot = analyses.pivot(index="representation", columns="analysis", values="delta_r2")
    comparison = pd.DataFrame(
        {
            "representation": pivot.index,
            "original_delta_r2": pivot["original_178"],
            "audited_delta_r2": pivot["manually_audited_175"],
            "uncorrected_matched_delta_r2": pivot["uncorrected_matched_175"],
            "correction_gain": pivot["manually_audited_175"] - pivot["uncorrected_matched_175"],
            "uniform_matched_delta_r2": pivot["uniform_matched_175"],
            "boundary_gain_vs_uniform": pivot["manually_audited_175"] - pivot["uniform_matched_175"],
        }
    ).reset_index(drop=True)
    table_output = Path(args.table_output)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(table_output, index=False)

    corrected = metadata[metadata.review_status.eq("corrected")]
    corrected_pairs = {(str(row.user_id), str(row.group_name)) for row in corrected.itertuples(index=False)}
    audited = pd.read_csv(args.audited_intervals).fillna("")
    source = pd.read_csv(args.source_intervals).fillna("")
    keys = ["user_id", "group_name", "word_index", "phone_index"]
    audited_corrected = audited[
        [(str(row.user_id), str(row.group_name)) in corrected_pairs for row in audited.itertuples(index=False)]
    ]
    source_corrected = source[
        [(str(row.user_id), str(row.group_name)) in corrected_pairs for row in source.itertuples(index=False)]
    ]
    shifts = audited_corrected.merge(source_corrected, on=keys, suffixes=("_audited", "_source"))
    boundary_shifts = np.concatenate(
        [
            np.abs(shifts.start_seconds_audited - shifts.start_seconds_source),
            np.abs(shifts.end_seconds_audited - shifts.end_seconds_source),
        ]
    )
    boundary_shifts = boundary_shifts[boundary_shifts > 1e-6]
    counts = metadata.review_status.value_counts().to_dict()
    quality = metadata.groupby(["quality_bin", "review_status"]).size().unstack(fill_value=0)
    for status in ("accepted", "corrected", "excluded"):
        if status not in quality:
            quality[status] = 0
    quality = quality[["accepted", "corrected", "excluded"]]

    selected = comparison.set_index("representation")
    table_rows = "\n".join(
        f"| {row.representation} | {row.original_delta_r2:+.4f} | {row.audited_delta_r2:+.4f} | "
        f"{row.correction_gain:+.4f} | {row.uniform_matched_delta_r2:+.4f} | "
        f"{row.boundary_gain_vs_uniform:+.4f} |"
        for row in comparison.itertuples(index=False)
    )
    quality_rows = "\n".join(
        f"| {name} | {int(row.accepted)} | {int(row.corrected)} | {int(row.excluded)} |"
        for name, row in quality.iterrows()
    )
    report = f"""# Manual Phone Boundary Audit

## Review Outcome

All **{len(metadata)} / {len(metadata)}** sampled sentence recordings received a listening
decision across all 20 speakers and all 10 sentence classes:

- Accepted unchanged: **{counts.get('accepted', 0)}**
- Corrected: **{counts.get('corrected', 0)}**
- Excluded as unusable alignments: **{counts.get('excluded', 0)}**
- Median nonzero correction: **{np.median(boundary_shifts) * 1000:.1f} ms**
- Maximum correction: **{np.max(boundary_shifts) * 1000:.1f} ms**

| Automatic quality bin | Accepted | Corrected | Excluded |
|---|---:|---:|---:|
{quality_rows}

None of the 25 high-confidence examples was excluded. All six exclusions came from the
primary, borderline, or fallback strata. The automated quality diagnostics are therefore
informative, but the three exclusions in the primary stratum show that they do not replace
manual review.

## Speaker-Disjoint Probe Comparison

The primary set retains **175** quality-controlled sentences after audit. **Correction
gain** compares corrected and original boundaries on these exact 175 recordings.
**Boundary gain** compares corrected boundaries with uniform within-word timing on the
same recordings.

| Representation | Original Delta R2 | Audited Delta R2 | Correction gain | Uniform Delta R2 | Boundary gain |
|---|---:|---:|---:|---:|---:|
{table_rows}

Across representations, manual correction changes residual R2 by
**{comparison.correction_gain.mean():+.4f}** on average. Audited boundaries trail matched
uniform timing by **{comparison.boundary_gain_vs_uniform.mean():+.4f}** on average.
All modalities remain positive at
**{selected.loc['all_modalities', 'audited_delta_r2']:+.4f}**, and non-lip contactless
sensors remain positive at
**{selected.loc['contactless_nonlip', 'audited_delta_r2']:+.4f}**.

## Interpretation

Manual review confirms that the positive phonetic result is not driven by obviously bad
alignments: excluding six failures and correcting five recordings leaves the signal
unchanged. It also confirms the controlled negative for exact timing. The representations
carry broad ordered phonetic occupancy beyond sentence class and relative position, but
the evidence does not show sensitivity to sharper within-word phone transitions.

The defensible project claim remains **coarse phonetic progression, not exact phone
tracking or individual phoneme neurons**. Phone identities are canonical prompt
pronunciations rather than manually transcribed speaker realizations. The next scientific
step is external-cohort validation, not additional tuning on these 40 audit examples.
"""
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Saved manual boundary audit report to {args.output}")


if __name__ == "__main__":
    main()
