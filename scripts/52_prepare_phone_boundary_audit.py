#!/usr/bin/env python3
"""Prepare a balanced manual audit set for phone-CTC boundaries."""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.data.textgrid import TextGridInterval, write_textgrid


def _quality_bin(row: pd.Series) -> str:
    if float(row.fallback_word_fraction) > 0:
        return "fallback"
    if float(row.mean_token_probability) >= 0.15 and float(row.phone_error_rate) <= 0.5:
        return "high"
    if float(row.mean_token_probability) >= 0.05 and float(row.phone_error_rate) <= 0.75:
        return "primary"
    return "borderline"


def _select_balanced(frame: pd.DataFrame, per_class: int) -> pd.DataFrame:
    """Greedily cover each class while keeping speaker and quality counts balanced."""
    selected: list[int] = []
    speaker_counts: dict[str, int] = {}
    quality_counts: dict[str, int] = {}
    quality_target = {"high": 12, "primary": 12, "borderline": 8, "fallback": 8}
    for group_name in sorted(frame.group_name.unique(), key=lambda value: int(str(value).replace("sentences", ""))):
        candidates = frame[frame.group_name == group_name].copy()
        for _ in range(per_class):
            candidates = candidates[~candidates.index.isin(selected)]
            if candidates.empty:
                raise ValueError(f"Not enough candidates for {group_name}")

            def score(row: pd.Series) -> tuple[float, float, float, str]:
                speaker = str(row.user_id)
                quality = str(row.quality_bin)
                quality_overflow = max(0, quality_counts.get(quality, 0) + 1 - quality_target.get(quality, 0))
                target_gap = quality_counts.get(quality, 0) / max(quality_target.get(quality, 1), 1)
                return (speaker_counts.get(speaker, 0), quality_overflow, target_gap, speaker)

            chosen_index = min(candidates.index, key=lambda index: score(candidates.loc[index]))
            chosen = frame.loc[chosen_index]
            selected.append(chosen_index)
            speaker = str(chosen.user_id)
            quality = str(chosen.quality_bin)
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
            quality_counts[quality] = quality_counts.get(quality, 0) + 1
    return frame.loc[selected].copy()


def _audio_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as audio:
        return audio.getnframes() / audio.getframerate()


def _replace_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(target.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_pronunciations.csv")
    parser.add_argument("--phone-intervals", default="artifacts/forced_alignment/phone_ctc_intervals.csv")
    parser.add_argument("--phone-audit", default="reports/results/phone_ctc_alignment_audit.csv")
    parser.add_argument("--word-alignments", default="artifacts/forced_alignment/wav2vec2_word_alignments.csv")
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--output-dir", default="artifacts/phone_boundary_audit")
    parser.add_argument("--metadata-output", default="metadata/phone_boundary_audit_set.csv")
    parser.add_argument("--report-output", default="reports/phone_boundary_audit_plan.md")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    existing_reviews = list((output_dir / "reviews").glob("*.json"))
    if existing_reviews and not args.force:
        raise RuntimeError(
            f"{len(existing_reviews)} saved reviews already exist; use --force only to replace the audit package"
        )

    manifest = pd.read_csv(args.manifest).fillna("")
    manifest = manifest[manifest.prompt_type.eq("sentence") & manifest.audio_path.map(lambda value: Path(value).exists())]
    audit = pd.read_csv(args.phone_audit).fillna("")
    numeric = ["mean_token_probability", "phone_error_rate", "center_in_word_fraction", "fallback_word_fraction"]
    for column in numeric:
        audit[column] = pd.to_numeric(audit[column], errors="coerce").fillna(0.0)
    candidates = manifest.merge(audit, on=["user_id", "group_name"], how="inner", suffixes=("", "_audit"))
    candidates = candidates[candidates.status.eq("aligned")].copy()
    candidates["quality_bin"] = candidates.apply(_quality_bin, axis=1)
    selected = _select_balanced(candidates, args.per_class)

    phone_intervals = pd.read_csv(args.phone_intervals).fillna("")
    words = pd.read_csv(args.word_alignments).fillna("")
    phone_groups = {
        (str(user), str(group)): part.sort_values(["word_index", "phone_index"])
        for (user, group), part in phone_intervals.groupby(
            [phone_intervals.user_id.astype(str), phone_intervals.group_name.astype(str)]
        )
    }
    word_groups = {
        (str(user), str(group)): part.sort_values("word_index")
        for (user, group), part in words.groupby([words.user_id.astype(str), words.group_name.astype(str)])
    }
    items = []
    metadata_rows = []
    for order, row in enumerate(selected.itertuples(index=False), start=1):
        pair = (str(row.user_id), str(row.group_name))
        key = f"u{row.user_id}_{row.group_name}"
        audio_source = Path(row.audio_path)
        duration = _audio_duration(audio_source)
        audio_link = output_dir / "audio" / f"{key}.wav"
        textgrid_path = output_dir / "textgrids" / f"{key}.TextGrid"
        _replace_symlink(audio_link, audio_source)
        phone_frame = phone_groups[pair]
        word_frame = word_groups[pair]
        phone_tier = [
            TextGridInterval(float(item.start_seconds), float(item.end_seconds), str(item.ipa))
            for item in phone_frame.itertuples(index=False)
        ]
        word_tier = [
            TextGridInterval(float(item.start_seconds), float(item.end_seconds), str(item.word))
            for item in word_frame.itertuples(index=False)
        ]
        write_textgrid(textgrid_path, duration, {"words": word_tier, "phones": phone_tier})
        phones = [
            {
                "word_index": int(item.word_index),
                "phone_index": int(item.phone_index),
                "word": str(item.word),
                "arpabet": str(item.arpabet),
                "ipa": str(item.ipa),
                "start": float(item.start_seconds),
                "end": float(item.end_seconds),
                "original_start": float(item.start_seconds),
                "original_end": float(item.end_seconds),
                "confidence": float(item.mean_token_probability),
                "method": str(item.method),
            }
            for item in phone_frame.itertuples(index=False)
        ]
        words_json = [
            {"index": int(item.word_index), "text": str(item.word), "start": float(item.start_seconds),
             "end": float(item.end_seconds)}
            for item in word_frame.itertuples(index=False)
        ]
        item = {
            "key": key,
            "order": order,
            "user_id": str(row.user_id),
            "group_name": str(row.group_name),
            "class_id": int(row.class_id),
            "transcript": str(row.transcript),
            "quality_bin": str(row.quality_bin),
            "phone_error_rate": float(row.phone_error_rate),
            "mean_token_probability": float(row.mean_token_probability),
            "fallback_word_fraction": float(row.fallback_word_fraction),
            "duration": duration,
            "audio_url": f"/audit/audio/{key}.wav",
            "textgrid_path": str(textgrid_path),
            "words": words_json,
            "phones": phones,
            "review": {"status": "unreviewed", "notes": ""},
        }
        items.append(item)
        metadata_rows.append(
            {
                "order": order,
                "user_id": row.user_id,
                "group_name": row.group_name,
                "class_id": int(row.class_id),
                "transcript": row.transcript,
                "quality_bin": row.quality_bin,
                "phone_error_rate": float(row.phone_error_rate),
                "mean_token_probability": float(row.mean_token_probability),
                "fallback_word_fraction": float(row.fallback_word_fraction),
                "review_status": "unreviewed",
                "review_notes": "",
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "reviews").mkdir(exist_ok=True)
    (output_dir / "data.json").write_text(json.dumps({"items": items}, indent=2), encoding="utf-8")
    metadata = pd.DataFrame(metadata_rows)
    metadata_path = Path(args.metadata_output)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata.to_csv(metadata_path, index=False)
    quality_counts = metadata.quality_bin.value_counts().to_dict()
    speaker_counts = metadata.user_id.value_counts()
    report = f"""# Phone Boundary Manual Audit Plan

The audit set contains **{len(metadata)}** sentence recordings sampled deterministically
from the direct phone-CTC alignment. It covers all **{metadata.group_name.nunique()}**
sentence classes ({args.per_class} recordings each) and all **{metadata.user_id.nunique()}**
speakers ({speaker_counts.min()}-{speaker_counts.max()} recordings each).

## Sampling

- High-confidence alignments: **{quality_counts.get('high', 0)}**
- Primary-gate alignments: **{quality_counts.get('primary', 0)}**
- Borderline alignments: **{quality_counts.get('borderline', 0)}**
- Alignments containing uniform word fallback: **{quality_counts.get('fallback', 0)}**

The tracked manifest is `{args.metadata_output}`. Audio symlinks, editable TextGrids,
review JSON, and browser data remain under ignored `{args.output_dir}` because they contain
local paths or generated data.

## Workflow

1. Run `make phone-boundary-audit`.
2. Listen to each clip and drag any incorrect phone boundaries.
3. Mark every recording accepted, corrected, or excluded.
4. Run `make phone-boundary-import`; incomplete audits are rejected.

## Decision Rule

Listen to every clip and mark it accepted, corrected, or excluded. Corrected boundaries
must remain ordered, non-overlapping, and inside the recording. The automated phonetic
claims should only be rerun after all 40 rows have a manual decision. This audit tests
boundary validity; it does not turn canonical prompt pronunciations into observed
speaker-specific phonetic transcriptions.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(
        f"Prepared {len(metadata)} recordings; speakers={metadata.user_id.nunique()} "
        f"classes={metadata.group_name.nunique()} quality={quality_counts}"
    )


if __name__ == "__main__":
    main()
