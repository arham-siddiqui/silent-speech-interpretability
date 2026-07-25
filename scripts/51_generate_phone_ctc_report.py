#!/usr/bin/env python3
"""Generate tracked summaries for direct phone-CTC alignment and sparse linkage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _macro(path: str, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    output = frame.groupby("representation", as_index=False).agg(
        r2=("r2_mean", "mean"), baseline_r2=("class_position_baseline_r2_mean", "mean"),
        delta_r2=("delta_r2_mean", "mean"), correlation=("correlation_mean", "mean"),
        order_margin=("order_margin_mean", "mean"),
    )
    output.insert(0, "analysis", label)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main", default="reports/results/phone_ctc_sentence_probe_summary.csv")
    parser.add_argument("--uniform", default="reports/results/uniform_sentence_probe_summary_matched.csv")
    parser.add_argument("--lenient", default="reports/results/phone_ctc_sentence_probe_summary_lenient.csv")
    parser.add_argument("--strict", default="reports/results/phone_ctc_sentence_probe_summary_strict_cv.csv")
    parser.add_argument("--alignment-audit", default="reports/results/phone_ctc_alignment_audit.csv")
    parser.add_argument("--sparse-ablation", default="reports/results/phone_sparse_feature_ablation.csv")
    parser.add_argument("--sparse-rankings", default="reports/results/phone_sparse_feature_rankings.csv")
    parser.add_argument("--main-targets", default="artifacts/forced_alignment/phone_ctc_sentence_targets.npz")
    parser.add_argument("--lenient-targets", default="artifacts/forced_alignment/phone_ctc_sentence_targets_lenient.npz")
    parser.add_argument("--strict-targets", default="artifacts/forced_alignment/phone_ctc_sentence_targets_strict_cv.npz")
    parser.add_argument("--table-dir", default="reports/tables")
    parser.add_argument("--output", default="reports/phone_ctc_interpretability.md")
    args = parser.parse_args()

    main_summary = pd.read_csv(args.main)
    macro = pd.concat(
        [_macro(args.main, "direct_main"), _macro(args.uniform, "uniform_matched"),
         _macro(args.lenient, "direct_lenient"), _macro(args.strict, "direct_strict_cv")],
        ignore_index=True,
    )
    direct = macro[macro.analysis == "direct_main"].set_index("representation")
    uniform = macro[macro.analysis == "uniform_matched"].set_index("representation")
    macro["boundary_gain_vs_uniform"] = macro.apply(
        lambda row: row.delta_r2 - uniform.loc[row.representation, "delta_r2"]
        if row.analysis == "direct_main" else np.nan,
        axis=1,
    )
    alignment = pd.read_csv(args.alignment_audit).fillna("")
    nonvowel = alignment[~alignment.group_name.str.startswith("vowel")].copy()
    fallback_words = int(pd.to_numeric(nonvowel.fallback_word_count).sum())
    intervals = pd.read_csv("artifacts/forced_alignment/phone_ctc_intervals.csv")
    sparse = pd.read_csv(args.sparse_ablation)
    sparse_summary = sparse.groupby(["mode", "k"], as_index=False).agg(
        macro_r2=("macro_r2", "mean"), baseline_r2=("baseline_macro_r2", "mean"),
        delta_r2=("delta_r2", "mean"), ablation_loss=("ablation_loss", "mean"),
        random_mean_loss=("random_mean_loss", "mean"), mean_fold_p=("random_p_value", "mean"),
        content_overlap=("content_top_k_overlap", "mean"),
        significant_folds=("random_p_value", lambda values: int((values < 0.05).sum())),
    )
    rankings = pd.read_csv(args.sparse_rankings)
    target_counts = {
        "direct_main": len(np.load(args.main_targets)["values"]),
        "direct_lenient": len(np.load(args.lenient_targets)["values"]),
        "direct_strict_cv": len(np.load(args.strict_targets)["values"]),
    }
    rank_correlations = rankings.groupby("fold").apply(
        lambda frame: frame[["phone_rank", "content_rank"]].corr(method="spearman").iloc[0, 1],
        include_groups=False,
    )

    table_dir = Path(args.table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)
    main_summary.to_csv(table_dir / "phone_ctc_sentence_probe_summary.csv", index=False)
    macro.to_csv(table_dir / "phone_ctc_probe_sensitivity.csv", index=False)
    sparse_summary.to_csv(table_dir / "phone_sparse_feature_ablation_summary.csv", index=False)
    pd.DataFrame(
        [{"aligned_recordings": int((alignment.status == "aligned").sum()), "phone_intervals": len(intervals),
          "fallback_words": fallback_words, "main_sentence_targets": target_counts["direct_main"],
          "lenient_sentence_targets": target_counts["direct_lenient"],
          "strict_cv_sentence_targets": target_counts["direct_strict_cv"]}]
    ).to_csv(table_dir / "phone_ctc_alignment_summary.csv", index=False)

    main_rows = "\n".join(
        f"| {name} | {row.r2:.3f} | {row.baseline_r2:.3f} | {row.delta_r2:+.3f} | {row.order_margin:+.3f} |"
        for name, row in direct.sort_values("delta_r2", ascending=False).iterrows()
    )
    sensitivity_rows = "\n".join(
        f"| {analysis} | {target_counts[analysis]} | "
        f"{macro[(macro.analysis == analysis) & (macro.representation == 'all_modalities')].delta_r2.iloc[0]:+.3f} | "
        f"{macro[(macro.analysis == analysis) & (macro.representation == 'contactless_nonlip')].delta_r2.iloc[0]:+.3f} |"
        for analysis in ("direct_lenient", "direct_main", "direct_strict_cv")
    )
    best_features = main_summary.loc[main_summary.groupby("feature").delta_r2_mean.idxmax()].sort_values("feature")
    feature_rows = "\n".join(
        f"| {row.feature} | {row.representation} | {row.delta_r2_mean:+.3f} | {row.order_margin_mean:+.3f} |"
        for row in best_features.itertuples(index=False)
    )
    sparse_k50 = sparse_summary[(sparse_summary["mode"] == "phone_top_ablation") & (sparse_summary.k == 50)].iloc[0]
    direct_uniform_gain = direct.delta_r2 - uniform.delta_r2
    report = f"""# Direct Phone-CTC Interpretability

## Alignment Audit

The direct IPA phoneme recognizer aligned **{int((alignment.status == 'aligned').sum())} / {len(alignment)}**
paired recordings and produced **{len(intervals)}** canonical phone intervals. Only
**{fallback_words}** word spans required a marked uniform fallback after phone emissions
collapsed at a word anchor. The primary probe uses **{target_counts['direct_main']} / 198** sentence recordings across
all 20 speakers and all 10 sentence classes; short isolated words are excluded because
their phone-recognition quality is not balanced across classes.

## Speaker-Disjoint Sentence Probes

| Representation | R2 | Class + position baseline | Delta R2 | Order margin |
|---|---:|---:|---:|---:|
{main_rows}

The strongest primary result is all modalities at **{direct.loc['all_modalities', 'delta_r2']:+.3f}**
residual R2, followed by contactless non-lip sensors at
**{direct.loc['contactless_nonlip', 'delta_r2']:+.3f}**. Lip alone does not improve the
class/position baseline (**{direct.loc['lip', 'delta_r2']:+.3f}**).

## Quality Sensitivity

| Gate | Sentences | All modalities Delta R2 | Contactless Delta R2 |
|---|---:|---:|---:|
{sensitivity_rows}

The positive result strengthens under the strictest fold-valid gate, so it is not driven
by low-confidence alignments.

## Exact-Boundary Control

Using uniform phone subdivisions on the exact same 178 sentences gives
**{uniform.loc['all_modalities', 'delta_r2']:+.3f}** all-modality residual R2 versus
**{direct.loc['all_modalities', 'delta_r2']:+.3f}** with direct phone timing. The mean
direct-minus-uniform gain is **{direct_uniform_gain.mean():+.3f}** across representations.
Thus the experiment supports broad ordered phonetic occupancy, but does not show that the
sensor representations track the sharper within-word phone boundaries.

## Best Feature Families

| Phonetic family | Best representation | Delta R2 | Order margin |
|---|---|---:|---:|
{feature_rows}

## Sparse-Feature Linkage

The fold-local temporal HuBERT sparse codes add **{sparse_summary[(sparse_summary['mode']=='full')].delta_r2.iloc[0]:+.3f}**
macro R2 over class/position. Ablating the 50 phone-ranked features loses
**{sparse_k50.ablation_loss:.4f}** R2 versus **{sparse_k50.random_mean_loss:.4f}** for random
features, but only **{int(sparse_k50.significant_folds)} / 5** folds reach a within-fold
random-control p-value below 0.05. Their mean top-50 overlap with existing HuBERT content
features is **{sparse_k50.content_overlap:.1f}**, below the chance expectation of
**{50*50/512:.1f}**, and foldwise rank correlation averages **{rank_correlations.mean():+.3f}**.

This is a controlled negative: phone timing is weakly recoverable from the sparse code but
is not stably concentrated in the previously identified HuBERT-causal content features.
No individual sparse feature should be named as a phoneme from these results.
"""
    Path(args.output).write_text(report, encoding="utf-8")
    print(f"Saved direct phone-CTC report to {args.output}")


if __name__ == "__main__":
    main()
