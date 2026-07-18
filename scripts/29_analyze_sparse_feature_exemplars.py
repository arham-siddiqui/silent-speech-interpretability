#!/usr/bin/env python3
"""Validate ranked sparse features with their strongest held-out utterances."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from silent_speech_interpretability.interp.feature_ranking import eta_squared
from silent_speech_interpretability.interp.sae import encode_sae, load_sae


TYPE_NAMES = {0: "vowel", 1: "word", 2: "sentence"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", default="artifacts/activations/hubert_cv")
    parser.add_argument("--sae-dir", default="artifacts/sae/hubert_bottleneck")
    parser.add_argument("--rankings", default="reports/results/hubert_bottleneck_feature_rankings.csv")
    parser.add_argument("--features-per-fold", type=int, default=3)
    parser.add_argument("--exemplars-per-feature", type=int, default=5)
    parser.add_argument("--output", default="reports/results/hubert_sparse_feature_exemplars.csv")
    parser.add_argument("--summary-output", default="reports/results/hubert_sparse_feature_exemplar_summary.csv")
    parser.add_argument("--report-output", default="reports/hubert_sparse_feature_exemplars.md")
    args = parser.parse_args()

    rankings = pd.read_csv(args.rankings)
    examples: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for fold in range(5):
        data = np.load(Path(args.activations_dir) / f"fold_{fold}_activations.npz")
        sae, payload = load_sae(Path(args.sae_dir) / f"fold_{fold}_sae.pt", torch.device("cpu"))
        features = encode_sae(
            sae,
            data["test_bottleneck"],
            np.asarray(payload["input_mean"]),
            np.asarray(payload["input_std"]),
            torch.device("cpu"),
        )
        selected = rankings[(rankings["fold"] == fold) & rankings["valid"]].nsmallest(
            args.features_per_fold, "rank"
        )
        for ranking in selected.itertuples(index=False):
            values = features[:, int(ranking.feature)]
            active = np.flatnonzero(values > 1e-6)
            ordered = active[np.argsort(-values[active])] if len(active) else np.array([], dtype=int)
            validation_indices = ordered[: min(10, len(ordered))]
            summaries.append(
                {
                    "fold": fold,
                    "feature": int(ranking.feature),
                    "rank": int(ranking.rank),
                    "train_best_class": int(ranking.best_class),
                    "train_best_type": int(ranking.best_type),
                    "train_best_type_name": TYPE_NAMES[int(ranking.best_type)],
                    "test_activation_frequency": float(len(active) / len(values)),
                    "test_class_selectivity": float(eta_squared(values[:, None], data["test_labels"])[0]),
                    "test_type_selectivity": float(eta_squared(values[:, None], data["test_type_labels"])[0]),
                    "top_active_count": int(len(validation_indices)),
                    "top_class_confirmation": float(
                        np.mean(data["test_labels"][validation_indices] == int(ranking.best_class))
                    ) if len(validation_indices) else np.nan,
                    "top_type_confirmation": float(
                        np.mean(data["test_type_labels"][validation_indices] == int(ranking.best_type))
                    ) if len(validation_indices) else np.nan,
                    "top_distinct_speakers": int(len(np.unique(data["test_user_ids"][validation_indices]))),
                }
            )
            for exemplar_rank, index in enumerate(ordered[: args.exemplars_per_feature], start=1):
                examples.append(
                    {
                        "fold": fold,
                        "feature": int(ranking.feature),
                        "feature_rank": int(ranking.rank),
                        "exemplar_rank": exemplar_rank,
                        "activation": float(values[index]),
                        "user_id": str(data["test_user_ids"][index]),
                        "group_name": str(data["test_group_names"][index]),
                        "class_id": int(data["test_labels"][index]),
                        "type": TYPE_NAMES[int(data["test_type_labels"][index])],
                    }
                )
        print(f"EXEMPLAR_PROGRESS fold={fold} selected_features={len(selected)}", flush=True)

    summary = pd.DataFrame(summaries)
    exemplar_frame = pd.DataFrame(examples)
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    exemplar_frame.to_csv(output, index=False)
    summary.to_csv(summary_output, index=False)

    summary_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.feature)} | {int(row.train_best_class)} | "
        f"{row.train_best_type_name.title()} | {100*row.test_activation_frequency:.1f}% | "
        f"{row.test_class_selectivity:.3f} | {row.test_type_selectivity:.3f} | "
        f"{100*row.top_class_confirmation:.1f}% | {100*row.top_type_confirmation:.1f}% | "
        f"{int(row.top_distinct_speakers)} |"
        for row in summary.itertuples(index=False)
    )
    example_rows = "\n".join(
        f"| {int(row.fold)} | {int(row.feature)} | {int(row.exemplar_rank)} | {row.user_id} | "
        f"{row.group_name} | {row.type.title()} | {row.activation:.3f} |"
        for row in exemplar_frame[exemplar_frame["exemplar_rank"] <= 3].itertuples(index=False)
    )
    report = f"""# Held-Out Sparse Feature Exemplars

The three highest-ranked fold-local SAE features were evaluated on speakers excluded
from both student and SAE training. Rankings come from training speakers; all metrics
and utterance exemplars below come from held-out test speakers.

## Held-Out Confirmation

| Fold | Feature | Train Class | Train Type | Test Frequency | Test Class Eta2 | Test Type Eta2 | Top-10 Class Match | Top-10 Type Match | Top-10 Speakers |
|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|
{summary_rows}

Across the 15 features, mean held-out class selectivity is
**{summary.test_class_selectivity.mean():.3f}** and type selectivity is
**{summary.test_type_selectivity.mean():.3f}**. Their top-10 activations match the
training-selected class **{100*summary.top_class_confirmation.mean():.1f}%** of the time
and coarse type **{100*summary.top_type_confirmation.mean():.1f}%** of the time. The top active utterances cover
**{summary.top_distinct_speakers.mean():.1f} distinct held-out speakers** on average,
which argues against single-speaker exemplars driving the ranking.

## Strongest Held-Out Utterances

| Fold | Feature | Exemplar | Speaker | Utterance | Type | Activation |
|---:|---:|---:|---|---|---|---:|
{example_rows}

## Interpretation Boundary

Group names make the class and coarse utterance type inspectable, but they do not provide
phoneme timestamps. These examples validate repeatable utterance-level preferences; they
do not justify assigning a phoneme or articulator name to an individual feature.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(f"Saved held-out exemplar report to {args.report_output}")


if __name__ == "__main__":
    main()
