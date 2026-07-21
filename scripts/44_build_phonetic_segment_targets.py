#!/usr/bin/env python3
"""Build four-segment phonetic occupancy targets from audited audio alignments."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import pandas as pd

from silent_speech_interpretability.data.phonetics import MANNER_FEATURES, interval_occupancy, phone_manner


def _trim_bounds(audio_path: str) -> tuple[float, float]:
    waveform, _ = librosa.load(audio_path, sr=16_000, mono=True)
    _trimmed, bounds = librosa.effects.trim(waveform, top_db=30)
    return bounds[0] / 16_000, bounds[1] / 16_000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_pronunciations.csv")
    parser.add_argument("--word-alignments", default="artifacts/forced_alignment/wav2vec2_word_alignments.csv")
    parser.add_argument("--alignment-audit", default="reports/results/wav2vec2_forced_alignment_audit.csv")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--output", default="artifacts/forced_alignment/phonetic_segment_targets.npz")
    parser.add_argument("--interval-output", default="artifacts/forced_alignment/interpolated_phone_intervals.csv")
    parser.add_argument("--report-output", default="reports/phonetic_target_audit.md")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna("")
    manifest = manifest[manifest.audio_path.map(lambda value: bool(value) and Path(value).exists())]
    words = pd.read_csv(args.word_alignments).fillna("")
    audit = pd.read_csv(args.alignment_audit).fillna("")
    audit_index = {(str(row.user_id), str(row.group_name)): row for row in audit.itertuples(index=False)}
    words_by_pair = {
        (str(user), str(group)): frame.sort_values("word_index")
        for (user, group), frame in words.groupby([words.user_id.astype(str), words.group_name.astype(str)])
    }

    targets, pairs, labels, confidences, methods, phone_rows = [], [], [], [], [], []
    for row in manifest.itertuples(index=False):
        pair = (str(row.user_id), str(row.group_name))
        intervals = []
        if row.prompt_type == "vowel":
            start, end = _trim_bounds(row.audio_path)
            intervals.append((start, end, phone_manner(row.vowel_arpabet)))
            confidence = 1.0
            method = "known_vowel_trimmed_interval"
            phone_rows.append(
                {"user_id": row.user_id, "group_name": row.group_name, "word": row.transcript, "phone": row.vowel_arpabet,
                 "start_seconds": start, "end_seconds": end, "method": method}
            )
        else:
            aligned = audit_index[pair]
            start, end = float(aligned.trim_start_seconds), float(aligned.trim_end_seconds)
            confidence = float(aligned.mean_token_probability)
            method = "ctc_word_aligned_phone_uniform_interpolation"
            for word in words_by_pair[pair].itertuples(index=False):
                phones = str(word.arpabet).split()
                boundaries = np.linspace(float(word.start_seconds), float(word.end_seconds), len(phones) + 1)
                for index, phone in enumerate(phones):
                    manner = phone_manner(phone)
                    intervals.append((float(boundaries[index]), float(boundaries[index + 1]), manner))
                    phone_rows.append(
                        {"user_id": row.user_id, "group_name": row.group_name, "word": word.word, "phone": phone,
                         "start_seconds": boundaries[index], "end_seconds": boundaries[index + 1], "method": method}
                    )
        targets.append(interval_occupancy(intervals, start, end, args.segments))
        pairs.append(pair)
        labels.append(int(row.class_id))
        confidences.append(confidence)
        methods.append(method)

    values = np.stack(targets).astype(np.float32)
    output = Path(args.output)
    interval_output = Path(args.interval_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        values=values,
        feature_names=np.asarray(MANNER_FEATURES),
        user_ids=np.asarray([pair[0] for pair in pairs]),
        group_names=np.asarray([pair[1] for pair in pairs]),
        labels=np.asarray(labels, dtype=np.int64),
        alignment_confidence=np.asarray(confidences, dtype=np.float32),
        alignment_method=np.asarray(methods),
        num_segments=np.asarray(args.segments),
    )
    pd.DataFrame(phone_rows).to_csv(interval_output, index=False)
    report = f"""# Phonetic Target Audit

- Usable paired recordings: **{len(values)}**.
- Relative temporal segments: **{args.segments}**.
- Broad phonetic features: **{len(MANNER_FEATURES)}** ({', '.join(MANNER_FEATURES)}).
- CTC word-aligned recordings: **{sum(method.startswith('ctc_') for method in methods)}**.
- Known isolated-vowel recordings: **{sum(method.startswith('known_') for method in methods)}**.
- Recordings meeting the main confidence cutoff (>=0.05): **{sum(value >= 0.05 for value in confidences)}**.

Word boundaries are constrained Viterbi alignments from `facebook/wav2vec2-base-960h`.
ARPAbet phones are distributed uniformly inside each aligned word interval. These are
therefore **interpolated phone occupancy targets**, not acoustically resolved phone
boundaries. The main probes retain only targets with alignment confidence >=0.05; isolated
vowels use the known vowel identity over the silence-trimmed acoustic interval.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(f"Saved {len(values)} phonetic segment targets to {output}")


if __name__ == "__main__":
    main()
