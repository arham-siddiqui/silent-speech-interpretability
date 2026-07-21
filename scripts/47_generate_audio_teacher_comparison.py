#!/usr/bin/env python3
"""Compare matched HuBERT and Wav2Vec2 temporal sensor students."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hubert", default="reports/results/temporal_sensor_student_cv.csv")
    parser.add_argument("--wav2vec2", default="reports/results/wav2vec2_temporal_sensor_student_cv.csv")
    parser.add_argument("--table-output", default="reports/tables/audio_teacher_comparison.csv")
    parser.add_argument("--report-output", default="reports/audio_teacher_comparison.md")
    args = parser.parse_args()

    rows = []
    for teacher, path in (("HuBERT", args.hubert), ("Wav2Vec2", args.wav2vec2)):
        frame = pd.read_csv(path)
        rows.append(
            {
                "teacher": teacher,
                "folds": int(frame.fold.nunique()),
                "accuracy_mean": frame.accuracy.mean(),
                "accuracy_std": frame.accuracy.std(ddof=1),
                "segment_cosine_mean": frame.segment_cosine.mean(),
                "reversed_cosine_mean": frame.reversed_segment_cosine.mean(),
                "order_margin_mean": frame.order_margin_reversed.mean(),
                "target_mse_mean": frame.target_mse.mean(),
            }
        )
    comparison = pd.DataFrame(rows)
    table_output = Path(args.table_output)
    table_output.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(table_output, index=False)
    hubert, wav2vec2 = comparison.iloc[0], comparison.iloc[1]
    accuracy_delta = 100 * (wav2vec2.accuracy_mean - hubert.accuracy_mean)
    cosine_delta = wav2vec2.segment_cosine_mean - hubert.segment_cosine_mean
    report = f"""# Audio Teacher Comparison

HuBERT and Wav2Vec2 were compared with the same four-segment pooling, silent-sensor
activations, student architecture, optimization, and five encoder/speaker-disjoint folds.

| Teacher | Accuracy | Segment cosine | Reversed cosine | Order margin | Target MSE |
|---|---:|---:|---:|---:|---:|
| HuBERT | {100*hubert.accuracy_mean:.1f}% | {hubert.segment_cosine_mean:.3f} | {hubert.reversed_cosine_mean:.3f} | {hubert.order_margin_mean:+.3f} | {hubert.target_mse_mean:.3f} |
| Wav2Vec2 | {100*wav2vec2.accuracy_mean:.1f}% | {wav2vec2.segment_cosine_mean:.3f} | {wav2vec2.reversed_cosine_mean:.3f} | {wav2vec2.order_margin_mean:+.3f} | {wav2vec2.target_mse_mean:.3f} |

Wav2Vec2 changes accuracy by **{accuracy_delta:+.1f} percentage points** and true-order
cosine by **{cosine_delta:+.3f}**. Because it is worse on
both objectives and has a substantially smaller order margin, HuBERT remains the selected
audio teacher. A Wav2Vec2 multitask hyperparameter sweep is not promoted: the matched base
experiment already rejects the hypothesis that this alternate teacher improves temporal
transfer under the current architecture.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(f"Saved audio teacher comparison to {args.report_output}")


if __name__ == "__main__":
    main()
