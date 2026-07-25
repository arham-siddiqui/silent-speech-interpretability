#!/usr/bin/env python3
"""Import completed phone-boundary reviews into a downstream interval table."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.data.textgrid import read_textgrid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-dir", default="artifacts/phone_boundary_audit")
    parser.add_argument("--metadata", default="metadata/phone_boundary_audit_set.csv")
    parser.add_argument("--source", default="artifacts/forced_alignment/phone_ctc_intervals.csv")
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--output", default="artifacts/forced_alignment/phone_ctc_intervals_audited.csv")
    parser.add_argument("--report-output", default="reports/phone_boundary_audit_results.md")
    args = parser.parse_args()

    metadata = pd.read_csv(args.metadata).fillna("")
    incomplete = metadata[metadata.review_status.eq("unreviewed")]
    if len(incomplete) and not args.allow_incomplete:
        raise RuntimeError(
            f"Manual audit is incomplete: {len(incomplete)} / {len(metadata)} recordings remain unreviewed"
        )
    source = pd.read_csv(args.source).fillna("")
    source["boundary_review_status"] = "not_audited"
    source["boundary_review_notes"] = ""
    review_dir = Path(args.audit_dir) / "reviews"
    shifts = []
    for review_path in sorted(review_dir.glob("*.json")):
        review = json.loads(review_path.read_text(encoding="utf-8"))
        reviewed_textgrid = Path(args.audit_dir) / "reviewed_textgrids" / f"{review['key']}.TextGrid"
        if reviewed_textgrid.exists() and review["status"] != "excluded":
            tiers = read_textgrid(reviewed_textgrid)
            textgrid_phones = tiers.get("phones", [])
            if len(textgrid_phones) != len(review["phones"]):
                raise ValueError(f"Reviewed TextGrid phone count changed for {review['key']}")
            for phone, interval in zip(review["phones"], textgrid_phones, strict=True):
                if str(phone["ipa"]) != interval.text:
                    raise ValueError(f"Reviewed TextGrid phone labels changed for {review['key']}")
                phone["start"] = interval.start
                phone["end"] = interval.end
        match = (
            source.user_id.astype(str).eq(str(review["user_id"]))
            & source.group_name.astype(str).eq(str(review["group_name"]))
        )
        indexes = source.index[match].tolist()
        if len(indexes) != len(review["phones"]):
            raise ValueError(f"Phone count changed for {review['key']}")
        if review["status"] == "excluded":
            source = source.loc[~match].copy()
            continue
        for index, phone in zip(indexes, review["phones"], strict=True):
            row = source.loc[index]
            if str(row.arpabet) != str(phone["arpabet"]):
                raise ValueError(f"Phone sequence changed for {review['key']}")
            old_start, old_end = float(row.start_seconds), float(row.end_seconds)
            new_start, new_end = float(phone["start"]), float(phone["end"])
            source.loc[index, "start_seconds"] = new_start
            source.loc[index, "end_seconds"] = new_end
            source.loc[index, "boundary_review_status"] = review["status"]
            source.loc[index, "boundary_review_notes"] = review.get("notes", "")
            shifts.extend([abs(new_start - old_start), abs(new_end - old_end)])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source.to_csv(output, index=False)

    counts = metadata.review_status.value_counts().to_dict()
    nonzero = np.asarray([shift for shift in shifts if shift > 1e-6])
    shift_summary = (
        f"Median nonzero boundary correction: **{np.median(nonzero) * 1000:.1f} ms**; "
        f"maximum: **{np.max(nonzero) * 1000:.1f} ms**."
        if len(nonzero)
        else "No reviewed boundary was moved."
    )
    report = f"""# Phone Boundary Audit Results

- Audit decisions: **{len(metadata) - len(incomplete)} / {len(metadata)}**
- Accepted without edits: **{counts.get('accepted', 0)}**
- Corrected: **{counts.get('corrected', 0)}**
- Excluded: **{counts.get('excluded', 0)}**
- Unreviewed: **{counts.get('unreviewed', 0)}**
- Output intervals: **{len(source)}**

{shift_summary}

This audit evaluates timing for a balanced sentence subset. Phone identities still come
from canonical prompt pronunciations and must not be described as manually transcribed
speaker realizations.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(f"Imported {len(metadata) - len(incomplete)} audit decisions into {output}")


if __name__ == "__main__":
    main()
