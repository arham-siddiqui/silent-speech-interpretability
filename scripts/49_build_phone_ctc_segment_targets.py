#!/usr/bin/env python3
"""Build quality-controlled sentence targets from phone-CTC intervals."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.data.phonetics import MANNER_FEATURES, interval_occupancy, phone_manner


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_pronunciations.csv")
    parser.add_argument("--phone-intervals", default="artifacts/forced_alignment/phone_ctc_intervals.csv")
    parser.add_argument("--phone-audit", default="reports/results/phone_ctc_alignment_audit.csv")
    parser.add_argument("--word-audit", default="reports/results/wav2vec2_forced_alignment_audit.csv")
    parser.add_argument("--prompt-type", default="sentence")
    parser.add_argument("--minimum-confidence", type=float, default=0.05)
    parser.add_argument("--maximum-phone-error", type=float, default=0.75)
    parser.add_argument("--minimum-center-in-word", type=float, default=0.5)
    parser.add_argument("--maximum-fallback-fraction", type=float, default=0.25)
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--output", default="artifacts/forced_alignment/phone_ctc_sentence_targets.npz")
    parser.add_argument("--report-output", default="reports/phone_ctc_target_audit.md")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna("")
    manifest = manifest[manifest.prompt_type == args.prompt_type]
    intervals = pd.read_csv(args.phone_intervals).fillna("")
    phone_column = "arpabet" if "arpabet" in intervals.columns else "phone"
    interval_groups = {
        (str(user), str(group)): frame.sort_values(["word_index", "phone_index"] if "phone_index" in frame else ["start_seconds"])
        for (user, group), frame in intervals.groupby([intervals.user_id.astype(str), intervals.group_name.astype(str)])
    }
    audit = pd.read_csv(args.phone_audit).fillna("")
    numeric = ["mean_token_probability", "phone_error_rate", "center_in_word_fraction", "fallback_word_fraction"]
    for column in numeric:
        audit[column] = pd.to_numeric(audit[column], errors="coerce")
    audit["quality_pass"] = (
        audit.status.eq("aligned")
        & audit.mean_token_probability.ge(args.minimum_confidence)
        & audit.phone_error_rate.le(args.maximum_phone_error)
        & audit.center_in_word_fraction.ge(args.minimum_center_in_word)
        & audit.fallback_word_fraction.le(args.maximum_fallback_fraction)
    )
    audit_index = {(str(row.user_id), str(row.group_name)): row for row in audit.itertuples(index=False)}
    word_audit = pd.read_csv(args.word_audit).fillna("")
    word_audit_index = {(str(row.user_id), str(row.group_name)): row for row in word_audit.itertuples(index=False)}

    values, pairs, labels, confidence, per, center_fraction, fallback_fraction = [], [], [], [], [], [], []
    method_counts: dict[str, int] = {}
    for row in manifest.itertuples(index=False):
        pair = (str(row.user_id), str(row.group_name))
        phone_quality = audit_index.get(pair)
        if phone_quality is None or not bool(phone_quality.quality_pass) or pair not in interval_groups:
            continue
        timing = word_audit_index[pair]
        frame = interval_groups[pair]
        phone_intervals = []
        for item in frame.itertuples(index=False):
            method = str(getattr(item, "method", "unknown"))
            method_counts[method] = method_counts.get(method, 0) + 1
            phone_intervals.append(
                (float(item.start_seconds), float(item.end_seconds), phone_manner(str(getattr(item, phone_column))))
            )
        values.append(
            interval_occupancy(
                phone_intervals,
                float(timing.trim_start_seconds),
                float(timing.trim_end_seconds),
                args.segments,
            )
        )
        pairs.append(pair)
        labels.append(int(row.class_id))
        confidence.append(float(phone_quality.mean_token_probability))
        per.append(float(phone_quality.phone_error_rate))
        center_fraction.append(float(phone_quality.center_in_word_fraction))
        fallback_fraction.append(float(phone_quality.fallback_word_fraction))

    target_values = np.stack(values).astype(np.float32)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        values=target_values,
        feature_names=np.asarray(MANNER_FEATURES),
        user_ids=np.asarray([pair[0] for pair in pairs]),
        group_names=np.asarray([pair[1] for pair in pairs]),
        labels=np.asarray(labels, dtype=np.int64),
        alignment_confidence=np.asarray(confidence, dtype=np.float32),
        phone_error_rate=np.asarray(per, dtype=np.float32),
        center_in_word_fraction=np.asarray(center_fraction, dtype=np.float32),
        fallback_word_fraction=np.asarray(fallback_fraction, dtype=np.float32),
        num_segments=np.asarray(args.segments),
    )
    coverage = pd.DataFrame({"user_id": [pair[0] for pair in pairs], "group_name": [pair[1] for pair in pairs]})
    report = f"""# Phone-CTC Target Audit

- Prompt subset: **{args.prompt_type}**.
- Quality-controlled recordings: **{len(values)} / {len(manifest)}**.
- Speakers retained: **{coverage.user_id.nunique()}**.
- Classes retained: **{coverage.group_name.nunique()}**.
- Minimum recordings per retained class: **{coverage.groupby('group_name').size().min()}**.
- Minimum recordings per retained speaker: **{coverage.groupby('user_id').size().min()}**.
- Median phone-CTC probability: **{np.median(confidence):.3f}**.
- Median unconstrained phone error rate: **{np.median(per):.3f}**.

## Quality Gate

- Forced-path probability >= {args.minimum_confidence}
- Phone error rate <= {args.maximum_phone_error}
- Phone emissions inside their word anchors >= {args.minimum_center_in_word:.2f}
- Uniform fallback words <= {args.maximum_fallback_fraction:.2f} of the utterance

Intervals are derived from direct IPA phone-CTC emission centers and midpoint boundaries
inside independently aligned words. Interval methods represented: {method_counts}.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(f"Saved {len(values)} quality-controlled phone targets to {output}")


if __name__ == "__main__":
    main()
