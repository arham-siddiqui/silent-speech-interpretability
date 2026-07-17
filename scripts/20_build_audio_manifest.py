#!/usr/bin/env python3
"""Create an audited manifest with RVTALL audio paths."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.audio_manifest import attach_rvtall_audio_paths


def _write_report(path: Path, audit: dict[str, object], manifest_path: Path) -> None:
    missing = audit["missing_pairs"]
    missing_text = ", ".join(f"`{pair}`" for pair in missing) if missing else "None"
    coverage = 100.0 * float(audit["coverage_fraction"])
    text = f"""# RVTALL Audio Manifest Audit

The local RVTALL audio discovery and alignment step completed successfully.

## Coverage

| Metric | Value |
|---|---:|
| Manifest rows | {audit['manifest_rows']} |
| WAV files discovered | {audit['audio_files_discovered']} |
| Unique speaker/group audio pairs | {audit['unique_audio_pairs_discovered']} |
| Rows with an audio path | {audit['rows_with_audio_path']} |
| Coverage | {coverage:.1f}% |
| Exact Kinect repetition matches | {audit['exact_reference_matches']} |
| Latest-repetition fallbacks | {audit['fallback_latest_matches']} |
| Missing pairs | {audit['missing_pair_count']} |

Missing pairs: {missing_text}

## Alignment Rule

Each manifest pair is matched to the WAV repetition synchronized with the lip
embedding selected by the repository's current duplicate-pair index. Concretely,
`video_N` maps to `audio_proc_N.wav`. A latest-available repetition is used only
when a pair has audio but no reference repetition.

The generated manifest is `{manifest_path}`. It contains machine-local absolute
paths and is intentionally excluded from Git; this audit report is pushable.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output", default="artifacts/manifest_with_audio.csv")
    parser.add_argument("--reference-npz", default=None)
    parser.add_argument("--audit-output", default="reports/results/audio_manifest_audit.json")
    parser.add_argument("--report-output", default="reports/audio_manifest_audit.md")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    rvtall_base = data_config.get("rvtall_base")
    if not rvtall_base:
        raise RuntimeError("Set data.rvtall_base in config before building the audio manifest.")

    input_path = Path(args.manifest or data_config.get("manifest_path", "artifacts/manifest.csv"))
    output_path = Path(args.output)
    audit_path = Path(args.audit_output)
    report_path = Path(args.report_output)
    reference_npz = args.reference_npz or data_config.get("embedding_paths", {}).get("lip")

    manifest = pd.read_csv(input_path)
    aligned, audit = attach_rvtall_audio_paths(manifest, rvtall_base, reference_npz)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(output_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    _write_report(report_path, audit, output_path)

    print(f"Wrote {output_path} ({audit['rows_with_audio_path']}/{audit['manifest_rows']} rows covered)")
    print(f"Wrote {audit_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
