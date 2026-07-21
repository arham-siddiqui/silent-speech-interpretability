#!/usr/bin/env python3
"""Attach authoritative speaker-specific RVTALL prompts to the audio manifest."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.data.prompts import SOURCE_URL, expected_prompt_rows, prompt_for


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_audio.csv")
    parser.add_argument("--output", default="artifacts/manifest_with_transcripts.csv")
    parser.add_argument("--prompt-map-output", default="metadata/rvtall_prompt_map.csv")
    parser.add_argument("--report-output", default="reports/rvtall_prompt_audit.md")
    parser.add_argument("--cohort-map", default="metadata/rvtall_audio_prompt_cohorts.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    cohort_path = Path(args.cohort_map)
    cohort_map = {}
    if cohort_path.exists():
        cohorts = pd.read_csv(cohort_path)
        cohort_map = dict(zip(cohorts.user_id.astype(int), cohorts.inferred_cohort.astype(str), strict=True))
    prompt_values = pd.DataFrame(
        [
            prompt_for(row.user_id, row.group_name, cohort_override=cohort_map.get(int(row.user_id)))
            for row in manifest.itertuples(index=False)
        ]
    )
    enriched = pd.concat([manifest.reset_index(drop=True), prompt_values], axis=1)
    expected = pd.DataFrame(expected_prompt_rows())
    observed_pairs = set(zip(enriched.user_id.astype(int), enriched.group_name.astype(str), strict=True))
    expected_pairs = set(zip(expected.user_id, expected.group_name, strict=True))
    missing = sorted(expected_pairs - observed_pairs)
    unexpected = sorted(observed_pairs - expected_pairs)
    variants = (
        expected.groupby(["group_name", "prompt_cohort", "transcript"], as_index=False)
        .size()
        .sort_values(["group_name", "prompt_cohort"])
    )

    output = Path(args.output)
    prompt_map_output = Path(args.prompt_map_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    prompt_map_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    expected.to_csv(prompt_map_output, index=False)

    variant_rows = "\n".join(
        f"| {row.group_name} | {row.prompt_cohort} | {row.transcript} | {int(row['size'])} |"
        for _, row in variants[variants.group_name.isin(["sentences7", "sentences9", "sentences10"])].iterrows()
    )
    missing_text = ", ".join(f"user {user} / {group}" for user, group in missing) or "None"
    unexpected_text = ", ".join(f"user {user} / {group}" for user, group in unexpected) or "None"
    report = f"""# RVTALL Prompt Audit

The anonymous RVTALL corpus labels are mapped to the prompts published in Table 5 of the
dataset paper: {SOURCE_URL}

## Coverage

- Expected speaker/group prompts: **{len(expected)}**.
- Audio-manifest rows enriched: **{len(enriched)}**.
- Missing expected pairs: **{len(missing)}** ({missing_text}).
- Unexpected pairs: **{len(unexpected)}** ({unexpected_text}).
- Rows with an existing audio file: **{enriched.audio_path.fillna('').map(lambda value: bool(value) and Path(value).exists()).sum()}**.
- Local audio cohort overrides loaded: **{len(cohort_map)}**.

## Speaker-Specific Sentence Classes

The publication assigns different text to sentence indices 7, 9, and 10 for three
participant cohorts. These must not be treated as one transcript per class.

The source table's participant IDs are preserved in the canonical mapping. When
`metadata/rvtall_audio_prompt_cohorts.csv` exists, the working manifest uses its audited
audio-folder mapping because local processed folder IDs do not match the published IDs.

| Group | Cohort | Transcript | Speakers |
|---|---|---|---:|
{variant_rows}

The paper prints `failling`; the normalized alignment transcript uses `failing` while
preserving the intended spoken phrase. Vowels retain their published IPA category and an
explicit ARPAbet target for later alignment.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(f"Saved transcript manifest to {output}")
    print(f"Saved prompt audit to {args.report_output}")


if __name__ == "__main__":
    main()
