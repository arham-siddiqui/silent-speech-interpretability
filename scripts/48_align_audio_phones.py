#!/usr/bin/env python3
"""Align canonical IPA phones with a direct phoneme-CTC acoustic model."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import pandas as pd
import torch

from silent_speech_interpretability.data.ctc_alignment import ctc_viterbi_align, sequence_error_rate
from silent_speech_interpretability.data.phonetics import arpabet_to_ipa


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _phone_words(word_pronunciations: str) -> list[dict[str, object]]:
    output = []
    for item in json.loads(word_pronunciations):
        arpabet = list(item["phones"])
        output.append({"word": str(item["word"]), "arpabet": arpabet, "ipa": [arpabet_to_ipa(phone) for phone in arpabet]})
    return output


def _anchored_phone_intervals(word_start: float, word_end: float, centers: list[float]) -> tuple[np.ndarray, bool]:
    if word_end <= word_start or not centers:
        raise ValueError("A positive word interval and phone centers are required")
    clipped = np.clip(np.asarray(centers, dtype=np.float64), word_start, word_end)
    internal = (clipped[:-1] + clipped[1:]) / 2.0
    boundaries = np.concatenate([[word_start], internal, [word_end]])
    if np.any(np.diff(boundaries) <= 1e-4):
        return np.linspace(word_start, word_end, len(centers) + 1), True
    return boundaries, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_pronunciations.csv")
    parser.add_argument("--word-alignments", default="artifacts/forced_alignment/wav2vec2_word_alignments.csv")
    parser.add_argument("--model-name", default="facebook/wav2vec2-lv-60-espeak-cv-ft")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-vowels", action="store_true")
    parser.add_argument("--output", default="artifacts/forced_alignment/phone_ctc_intervals.csv")
    parser.add_argument("--audit-output", default="reports/results/phone_ctc_alignment_audit.csv")
    args = parser.parse_args()

    from transformers import AutoFeatureExtractor, AutoModelForCTC, Wav2Vec2PhonemeCTCTokenizer

    manifest = pd.read_csv(args.manifest).fillna("")
    rows = manifest[manifest.audio_path.map(lambda value: bool(value) and Path(value).exists())].copy()
    if args.skip_vowels:
        rows = rows[rows.prompt_type != "vowel"]
    if args.limit is not None:
        rows = rows.head(args.limit)
    word_table = pd.read_csv(args.word_alignments).fillna("")
    word_groups = {
        (str(user), str(group)): frame.sort_values("word_index")
        for (user, group), frame in word_table.groupby([word_table.user_id.astype(str), word_table.group_name.astype(str)])
    }
    device = _device(args.device)
    tokenizer = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(
        args.model_name, do_phonemize=False, local_files_only=args.local_files_only
    )
    feature_extractor = AutoFeatureExtractor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForCTC.from_pretrained(args.model_name, local_files_only=args.local_files_only).to(device).eval()
    blank_id = int(tokenizer.pad_token_id)
    phone_rows, audit_rows = [], []
    started = time.perf_counter()

    for position, row in enumerate(rows.itertuples(index=False), start=1):
        pair = (str(row.user_id), str(row.group_name))
        audit = {"user_id": row.user_id, "group_name": row.group_name, "status": "error", "error": ""}
        row_phone_rows = []
        try:
            waveform, _ = librosa.load(row.audio_path, sr=16_000, mono=True)
            trimmed, bounds = librosa.effects.trim(waveform, top_db=30)
            trim_start, trim_end = bounds[0] / 16_000, bounds[1] / 16_000
            words = _phone_words(row.word_pronunciations)
            if row.prompt_type == "vowel":
                row_phone_rows.append(
                    {"user_id": row.user_id, "group_name": row.group_name, "class_id": row.class_id,
                     "word_index": 0, "phone_index": 0, "word": row.transcript,
                     "arpabet": words[0]["arpabet"][0], "ipa": words[0]["ipa"][0],
                     "start_seconds": trim_start, "end_seconds": trim_end,
                     "emission_center_seconds": (trim_start + trim_end) / 2,
                     "mean_token_probability": 1.0, "method": "known_vowel_trimmed_interval"}
                )
                audit.update({"status": "aligned", "phone_error_rate": 0.0, "mean_token_probability": 1.0,
                              "center_in_word_fraction": 1.0, "phone_count": 1, "trimmed_seconds": len(trimmed) / 16_000})
            else:
                target_ipa = [phone for word in words for phone in word["ipa"]]
                encoded = tokenizer(" ".join(target_ipa), add_special_tokens=False)
                target_ids = list(encoded.input_ids)
                encoded_tokens = tokenizer.convert_ids_to_tokens(target_ids)
                if encoded_tokens != target_ipa:
                    raise ValueError("IPA target does not round-trip through the phone tokenizer")
                inputs = feature_extractor(trimmed, sampling_rate=16_000, return_tensors="pt")
                with torch.no_grad():
                    logits = model(inputs.input_values.to(device)).logits[0]
                    log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
                spans = ctc_viterbi_align(log_probs, target_ids, blank_id)
                frame_seconds = len(trimmed) / 16_000 / len(log_probs)
                offset = trim_start
                centers = [offset + ((span.start_frame + span.end_frame) / 2) * frame_seconds for span in spans]
                hypothesis_ids = logits.argmax(dim=-1).cpu().unsqueeze(0)
                hypothesis = tokenizer.batch_decode(hypothesis_ids)[0].split()
                aligned_words = word_groups[pair]
                if len(aligned_words) != len(words):
                    raise ValueError("Phone target words do not match word alignment count")
                phone_offset = 0
                centers_inside = []
                durations = []
                fallback_words = 0
                for word_index, (word, aligned_word) in enumerate(zip(words, aligned_words.itertuples(index=False), strict=True)):
                    count = len(word["ipa"])
                    word_centers = centers[phone_offset : phone_offset + count]
                    boundaries, used_fallback = _anchored_phone_intervals(
                        float(aligned_word.start_seconds), float(aligned_word.end_seconds), word_centers
                    )
                    fallback_words += int(used_fallback)
                    for phone_index, (arpabet, ipa) in enumerate(zip(word["arpabet"], word["ipa"], strict=True)):
                        span = spans[phone_offset + phone_index]
                        center = word_centers[phone_index]
                        inside = float(aligned_word.start_seconds) <= center <= float(aligned_word.end_seconds)
                        centers_inside.append(inside)
                        durations.append(float(boundaries[phone_index + 1] - boundaries[phone_index]))
                        row_phone_rows.append(
                            {"user_id": row.user_id, "group_name": row.group_name, "class_id": row.class_id,
                             "word_index": word_index, "phone_index": phone_index, "word": word["word"],
                             "arpabet": arpabet, "ipa": ipa, "start_seconds": boundaries[phone_index],
                             "end_seconds": boundaries[phone_index + 1], "emission_center_seconds": center,
                             "mean_token_probability": float(np.exp(span.mean_log_probability)),
                             "method": "uniform_collapsed_word_fallback" if used_fallback else "phone_ctc_midpoints_with_word_anchors"}
                        )
                    phone_offset += count
                audit.update(
                    {"status": "aligned", "phone_error_rate": sequence_error_rate(target_ipa, hypothesis),
                     "mean_token_probability": float(np.exp(np.mean([span.mean_log_probability for span in spans]))),
                     "center_in_word_fraction": float(np.mean(centers_inside)), "phone_count": len(target_ipa),
                     "fallback_word_count": fallback_words, "fallback_word_fraction": fallback_words / len(words),
                     "median_phone_duration_seconds": float(np.median(durations)),
                     "minimum_phone_duration_seconds": float(np.min(durations)),
                     "maximum_phone_duration_seconds": float(np.max(durations)),
                     "trimmed_seconds": len(trimmed) / 16_000, "hypothesis": " ".join(hypothesis),
                     "reference": " ".join(target_ipa)}
                )
        except Exception as exc:
            audit["error"] = str(exc)
            row_phone_rows = []
        phone_rows.extend(row_phone_rows)
        audit_rows.append(audit)
        if position == 1 or position % 25 == 0 or position == len(rows):
            elapsed = time.perf_counter() - started
            remaining = elapsed / position * (len(rows) - position)
            print(
                f"PHONE_CTC progress={position}/{len(rows)} elapsed_seconds={elapsed:.1f} "
                f"estimated_remaining_seconds={remaining:.1f}", flush=True
            )

    output, audit_output = Path(args.output), Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(phone_rows).to_csv(output, index=False)
    pd.DataFrame(audit_rows).to_csv(audit_output, index=False)
    aligned = sum(row["status"] == "aligned" for row in audit_rows)
    print(f"Saved {aligned}/{len(audit_rows)} phone alignments to {output}")


if __name__ == "__main__":
    main()
