#!/usr/bin/env python3
"""Run multi-seed external generalization and radar-feature ablation experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRICS = [
    "accuracy",
    "segment_cosine",
    "order_margin_reversed",
    "residual_segment_cosine",
    "residual_order_margin_reversed",
]


def _run_student(
    *,
    protocol: str,
    feature_set: str,
    seeds: str,
    output: Path,
    max_epochs: int,
    residual_control: bool,
    device: str,
) -> None:
    command = [
        sys.executable,
        "scripts/59_run_external_radar_student.py",
        "--protocol",
        protocol,
        "--feature-set",
        feature_set,
        "--seeds",
        seeds,
        "--max-epochs",
        str(max_epochs),
        "--device",
        device,
        "--no-save-checkpoints",
        "--results-output",
        str(output),
        "--report-output",
        str(Path("artifacts/external/radar_command_words") / f"{output.stem}_draft.md"),
    ]
    if not residual_control:
        command.append("--no-residual-control")
    subprocess.run(command, check=True)


def _unit_bootstrap(
    data: pd.DataFrame,
    metric: str,
    unit_column: str,
    *,
    seed: int = 2026,
    draws: int = 20_000,
) -> tuple[float, float]:
    unit_means = data.groupby(unit_column)[metric].mean().dropna().to_numpy()
    if not len(unit_means):
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    sampled = rng.choice(unit_means, size=(draws, len(unit_means)), replace=True).mean(axis=1)
    return tuple(np.quantile(sampled, [0.025, 0.975]).tolist())


def _protocol_summary(data: pd.DataFrame, unit_column: str) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        low, high = _unit_bootstrap(data, metric, unit_column)
        seed_sd = data.groupby(unit_column)[metric].std(ddof=1).mean()
        rows.append(
            {
                "protocol": data["protocol"].iloc[0],
                "metric": metric,
                "mean": data[metric].mean(),
                "unit_bootstrap_95_low": low,
                "unit_bootstrap_95_high": high,
                "mean_within_unit_seed_sd": seed_sd,
                "held_out_units": data[unit_column].nunique(),
                "seeds": data["seed"].nunique(),
            }
        )
    return pd.DataFrame(rows)


def _format_interval(summary: pd.DataFrame, metric: str, percent: bool = False) -> str:
    row = summary[summary["metric"] == metric].iloc[0]
    if percent:
        return (
            f"{100 * row['mean']:.1f}% "
            f"[{100 * row['unit_bootstrap_95_low']:.1f}%, "
            f"{100 * row['unit_bootstrap_95_high']:.1f}%]"
        )
    return (
        f"{row['mean']:.3f} "
        f"[{row['unit_bootstrap_95_low']:.3f}, "
        f"{row['unit_bootstrap_95_high']:.3f}]"
    )


def _write_report(
    path: Path,
    session: pd.DataFrame,
    subject: pd.DataFrame,
    ablations: pd.DataFrame,
    summary: pd.DataFrame,
) -> None:
    session_summary = summary[summary["protocol"] == "session"]
    subject_summary = summary[summary["protocol"] == "subject"]
    ablation_summary = (
        ablations.groupby("feature_set")
        .agg(
            accuracy=("accuracy", "mean"),
            segment_cosine=("segment_cosine", "mean"),
            order_margin=("order_margin_reversed", "mean"),
        )
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    ablation_rows = "\n".join(
        f"| {row.feature_set} | {100 * row.accuracy:.1f}% | "
        f"{row.segment_cosine:.3f} | {row.order_margin:+.3f} |"
        for row in ablation_summary.itertuples(index=False)
    )
    subject_rows = "\n".join(
        f"| {row.test_user} | {100 * row.accuracy:.1f}% | {row.segment_cosine:.3f} | "
        f"{row.order_margin_reversed:+.3f} | {row.residual_segment_cosine:.3f} | "
        f"{row.residual_order_margin_reversed:+.3f} |"
        for row in subject.groupby("test_user", as_index=False).mean(numeric_only=True).itertuples(
            index=False
        )
    )
    report = f"""# External Radar Generalization And Ablation Batch

## Protocols

- Session-held-out: one session trains, one validates, and one tests, with both
  subjects represented in each split.
- Subject-held-out: two sessions from one subject train, that subject's third session
  validates, and all sessions from the other subject test.
- Reliability: {session['seed'].nunique()} optimization seeds per held-out unit.
- Intervals: empirical 95% bootstrap intervals over held-out sessions or subjects after
  averaging seeds. With only three sessions and two subjects, these intervals describe
  this corpus and are not population-level confidence intervals.

## Multi-seed results

| Protocol | Accuracy (95% interval) | HuBERT cosine | Order margin | Residual cosine | Residual order margin |
|---|---:|---:|---:|---:|---:|
| Session-held-out | {_format_interval(session_summary, "accuracy", True)} | {_format_interval(session_summary, "segment_cosine")} | {_format_interval(session_summary, "order_margin_reversed")} | {_format_interval(session_summary, "residual_segment_cosine")} | {_format_interval(session_summary, "residual_order_margin_reversed")} |
| Subject-held-out | {_format_interval(subject_summary, "accuracy", True)} | {_format_interval(subject_summary, "segment_cosine")} | {_format_interval(subject_summary, "order_margin_reversed")} | {_format_interval(subject_summary, "residual_segment_cosine")} | {_format_interval(subject_summary, "residual_order_margin_reversed")} |

Chance command accuracy is 2%.

## Optimization stability

- Session-held-out mean within-session seed SD: **{100 * session_summary.loc[session_summary['metric'] == 'accuracy', 'mean_within_unit_seed_sd'].iloc[0]:.1f} accuracy points** and **{session_summary.loc[session_summary['metric'] == 'segment_cosine', 'mean_within_unit_seed_sd'].iloc[0]:.3f} cosine**.
- Subject-held-out mean within-subject seed SD: **{100 * subject_summary.loc[subject_summary['metric'] == 'accuracy', 'mean_within_unit_seed_sd'].iloc[0]:.1f} accuracy points** and **{subject_summary.loc[subject_summary['metric'] == 'segment_cosine', 'mean_within_unit_seed_sd'].iloc[0]:.3f} cosine**.

## Subject transfer detail

| Held-out subject | Accuracy | HuBERT cosine | Order margin | Residual cosine | Residual order margin |
|---|---:|---:|---:|---:|---:|
{subject_rows}

## Radar feature ablations

Each ablation uses the established session-held-out protocol and seed 42. S12 and S32
denote the two scattering-parameter radargrams used by the source implementation.

| Feature set | Accuracy | HuBERT cosine | Order margin |
|---|---:|---:|---:|
{ablation_rows}

## Interpretation

Session transfer measures robustness to a new recording visit with familiar subjects.
Subject transfer is the stricter speaker-generalization test. The residual metrics remove
train-only command prototypes, so they test ordered exemplar variation beyond command
identity. Feature ablations identify whether performance depends more on radar channel
choice or on static magnitude versus temporal-change information.

The external corpus contains only two subjects, so its held-out-subject result should
be treated as a diagnostic boundary rather than a population-level estimate.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--device", default="mps")
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate summaries from existing raw result tables without retraining",
    )
    parser.add_argument(
        "--session-output",
        default="reports/tables/external_radar_session_multiseed.csv",
    )
    parser.add_argument(
        "--subject-output",
        default="reports/tables/external_radar_subject_multiseed.csv",
    )
    parser.add_argument(
        "--ablation-output",
        default="reports/tables/external_radar_feature_ablation.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/tables/external_radar_generalization_summary.csv",
    )
    parser.add_argument(
        "--report-output",
        default="reports/external_radar_generalization.md",
    )
    args = parser.parse_args()

    required = [
        Path("artifacts/external/radar_command_words/radar_temporal4_features.npz"),
        Path("artifacts/external/radar_command_words/hubert_temporal4_targets.npz"),
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing external artifacts: {', '.join(missing)}")

    session_output = Path(args.session_output)
    subject_output = Path(args.subject_output)
    if not args.report_only:
        _run_student(
            protocol="session",
            feature_set="all",
            seeds=args.seeds,
            output=session_output,
            max_epochs=args.max_epochs,
            residual_control=True,
            device=args.device,
        )
        _run_student(
            protocol="subject",
            feature_set="all",
            seeds=args.seeds,
            output=subject_output,
            max_epochs=args.max_epochs,
            residual_control=True,
            device=args.device,
        )

    session = pd.read_csv(session_output)
    subject = pd.read_csv(subject_output)
    ablation_output = Path(args.ablation_output)
    if args.report_only:
        ablations = pd.read_csv(ablation_output)
    else:
        ablation_parts = [session[session["seed"] == int(args.seeds.split(",")[0])].copy()]
        for feature_set in ("s12", "s32", "magnitude", "delta"):
            output = Path("artifacts/external/radar_command_words") / f"ablation_{feature_set}.csv"
            _run_student(
                protocol="session",
                feature_set=feature_set,
                seeds=args.seeds.split(",")[0],
                output=output,
                max_epochs=args.max_epochs,
                residual_control=False,
                device=args.device,
            )
            ablation_parts.append(pd.read_csv(output))
        ablations = pd.concat(ablation_parts, ignore_index=True)
        ablation_output.parent.mkdir(parents=True, exist_ok=True)
        ablations.to_csv(ablation_output, index=False)

    summary = pd.concat(
        [
            _protocol_summary(session, "test_session"),
            _protocol_summary(subject, "test_user"),
        ],
        ignore_index=True,
    )
    summary_output = Path(args.summary_output)
    summary_output.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_output, index=False)

    _write_report(
        Path(args.report_output),
        session,
        subject,
        ablations,
        summary,
    )
    print(f"Saved external validation batch report to {args.report_output}")


if __name__ == "__main__":
    main()
