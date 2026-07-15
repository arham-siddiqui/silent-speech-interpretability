#!/usr/bin/env python3
"""Audit whether reported evaluations are encoder-disjoint or only fusion-disjoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.configs import load_config


def _parse_speakers(value: str) -> set[int]:
    if pd.isna(value) or value == "":
        return set()
    return {int(part) for part in str(value).split(",") if part}


def _markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    shown = df[columns].copy().fillna("")
    rows = [[str(value) for value in row] for row in shown.to_numpy().tolist()]
    widths = [
        max(len(str(column)), *(len(row[idx]) for row in rows)) if rows else len(str(column))
        for idx, column in enumerate(columns)
    ]
    header = "| " + " | ".join(str(column).ljust(widths[idx]) for idx, column in enumerate(columns)) + " |"
    separator = "| " + " | ".join("-" * widths[idx] for idx in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(columns))) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    results_dir = Path(config["data"]["results_dir"])
    reports_dir = Path("reports")
    encoder_train = {int(speaker) for speaker in config.get("evaluation", {}).get("embedding_encoder_train_speakers", [])}
    fixed_test = {int(speaker) for speaker in config["splits"]["fixed_test_speakers"]}

    cv = pd.read_csv(results_dir / "speaker_cv_results.csv")
    fold_rows = []
    for fold, group in cv.groupby("fold"):
        test_speakers = _parse_speakers(group.iloc[0]["test_speakers"])
        overlap = sorted(test_speakers & encoder_train)
        fold_rows.append(
            {
                "fold": int(fold),
                "test_speakers": ",".join(map(str, sorted(test_speakers))),
                "encoder_seen_test_speakers": ",".join(map(str, overlap)),
                "num_encoder_seen_test_speakers": len(overlap),
                "encoder_disjoint_test": len(overlap) == 0,
            }
        )
    fold_audit = pd.DataFrame(fold_rows)

    fixed_overlap = sorted(fixed_test & encoder_train)
    valid_cv_folds = int(fold_audit["encoder_disjoint_test"].sum())
    audit = {
        "embedding_encoder_train_speakers": sorted(encoder_train),
        "fixed_test_speakers": sorted(fixed_test),
        "fixed_test_encoder_seen_speakers": fixed_overlap,
        "fixed_split_encoder_disjoint": len(fixed_overlap) == 0,
        "num_cv_folds": int(len(fold_audit)),
        "num_encoder_disjoint_cv_folds": valid_cv_folds,
        "cv_is_encoder_disjoint": valid_cv_folds == len(fold_audit),
        "interpretation": (
            "CV over precomputed embeddings is fusion-layer CV only when held-out speakers were used to train "
            "the encoders. Treat fixed split as the encoder-disjoint baseline unless encoders are retrained per fold."
        ),
    }
    (results_dir / "evaluation_sanity_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    fold_audit.to_csv(results_dir / "cv_encoder_overlap_audit.csv", index=False)

    report = [
        "# Evaluation Sanity Audit",
        "",
        "This audit checks whether evaluation splits are disjoint from the speakers used to train the precomputed encoders.",
        "",
        f"- Encoder training speakers: {sorted(encoder_train)}",
        f"- Fixed test speakers: {sorted(fixed_test)}",
        f"- Fixed split encoder-disjoint: {audit['fixed_split_encoder_disjoint']}",
        f"- Encoder-disjoint CV folds: {valid_cv_folds}/{len(fold_audit)}",
        "",
        "## CV Fold Overlap",
        "",
        _markdown_table(
            fold_audit,
            ["fold", "test_speakers", "encoder_seen_test_speakers", "num_encoder_seen_test_speakers", "encoder_disjoint_test"],
        ),
        "",
        "## Interpretation",
        "",
        audit["interpretation"],
        "",
    ]
    output = reports_dir / "evaluation_sanity_audit.md"
    output.write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote {output}")
    print(f"Encoder-disjoint CV folds: {valid_cv_folds}/{len(fold_audit)}")


if __name__ == "__main__":
    main()
