#!/usr/bin/env python3
"""Audit and manifest the external Wagner et al. radar command-word corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.data.external_radar import (
    discover_external_radar_corpus,
    read_radar_binary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--corpus-root",
        default="artifacts/external/radar_command_words/extracted/corpus",
    )
    parser.add_argument(
        "--manifest-output",
        default="artifacts/external/radar_command_words/manifest.csv",
    )
    parser.add_argument(
        "--summary-output",
        default="reports/tables/external_radar_corpus_summary.csv",
    )
    parser.add_argument(
        "--report-output",
        default="reports/external_radar_corpus_audit.md",
    )
    parser.add_argument("--inspect-files", type=int, default=12)
    args = parser.parse_args()

    manifest = discover_external_radar_corpus(args.corpus_root)
    expected_per_cell = (
        manifest.groupby(["user_id", "session_id", "class_id"]).size().rename("samples")
    )
    if expected_per_cell.nunique() != 1:
        raise RuntimeError("The external corpus is not balanced across subject/session/class cells")

    inspected = []
    if args.inspect_files > 0:
        sample_positions = pd.Series(range(len(manifest))).sample(
            min(args.inspect_files, len(manifest)), random_state=17
        )
        for position in sorted(sample_positions):
            row = manifest.iloc[position]
            recording = read_radar_binary(row["radar_path"])
            inspected.append(
                {
                    "sample_id": row["sample_id"],
                    "frames": recording.num_total_frames,
                    "frequency_steps": recording.num_steps,
                    "spectra": len(recording.radargrams),
                }
            )

    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)

    summary = (
        manifest.groupby(["user_id", "session_id"])
        .agg(samples=("sample_id", "size"), classes=("class_id", "nunique"))
        .reset_index()
    )
    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)

    frame_text = "not inspected"
    if inspected:
        inspection = pd.DataFrame(inspected)
        frame_text = (
            f"{inspection['frames'].min()}–{inspection['frames'].max()} frames; "
            f"{inspection['frequency_steps'].min()}–{inspection['frequency_steps'].max()} steps; "
            f"{inspection['spectra'].min()}–{inspection['spectra'].max()} spectra"
        )
    report = f"""# External Radar Corpus Audit

## Pairing result

- Samples: **{len(manifest):,}**
- Subjects: **{manifest['user_id'].nunique()}**
- Sessions: **{manifest['session_id'].nunique()}**
- Command classes: **{manifest['class_id'].nunique()}**
- Repetitions per subject/session/class: **{int(expected_per_cell.iloc[0])}**
- Random binary-file inspection ({len(inspected)} files): **{frame_text}**
- Pairing key: exact shared filename stem across radar, WAV audio, and text label files

Every manifest row has an existing radar file, paired audio file, and non-empty text label.
The manifest is sorted deterministically by subject, session, class, and repetition.

## Role in this project

This corpus provides an independent radar/audio test bed for the audio-teacher and
silent-sensor-student method. It changes the laboratory, language, speakers, command
inventory, and radar hardware relative to RVTALL. The planned evaluation holds out
recording sessions and uses audio only while creating teacher targets; inference uses
radar features alone.

## Scope limitation

The corpus has only two subjects. A positive result is therefore evidence of external
cross-session replication, not broad population-level speaker generalization.

## Provenance

- [Wagner et al., “Silent speech command word recognition using stepped frequency
  continuous wave radar,” *Scientific Reports* (2022)](https://pubmed.ncbi.nlm.nih.gov/35273225/).
- [Official supplementary implementation](https://github.com/TUD-STKS/radar_based_command_word_recognition)
  and corpus published by TU Dresden / VocalTractLab.
"""
    report_path = Path(args.report_output)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    print(f"Saved {len(manifest):,} paired samples to {manifest_path}")
    print(f"Saved audit report to {report_path}")


if __name__ == "__main__":
    main()
