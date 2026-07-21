#!/usr/bin/env python3
"""CTC-align known RVTALL prompts to their paired audio recordings."""

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

from silent_speech_interpretability.data.ctc_alignment import character_error_rate, ctc_viterbi_align


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _word_spans(tokens: list[str], spans, frame_seconds: float, offset_seconds: float) -> list[dict[str, float | str]]:
    words = []
    current_tokens = []
    current_spans = []
    for token, span in zip(tokens, spans, strict=True):
        if token == "|":
            if current_tokens:
                words.append(("".join(current_tokens), current_spans))
                current_tokens, current_spans = [], []
        else:
            current_tokens.append(token)
            current_spans.append(span)
    if current_tokens:
        words.append(("".join(current_tokens), current_spans))
    return [
        {
            "word": word.lower(),
            "start_seconds": offset_seconds + word_tokens[0].start_frame * frame_seconds,
            "end_seconds": offset_seconds + word_tokens[-1].end_frame * frame_seconds,
            "mean_token_probability": float(np.exp(np.mean([span.mean_log_probability for span in word_tokens]))),
        }
        for word, word_tokens in words
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_pronunciations.csv")
    parser.add_argument("--model-name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-vowels", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="artifacts/manifest_with_word_alignments.csv")
    parser.add_argument("--word-output", default="artifacts/forced_alignment/wav2vec2_word_alignments.csv")
    parser.add_argument("--audit-output", default="reports/results/wav2vec2_forced_alignment_audit.csv")
    args = parser.parse_args()

    from transformers import AutoModelForCTC, AutoProcessor

    manifest = pd.read_csv(args.manifest).fillna("")
    rows = manifest[manifest.audio_path.map(lambda value: bool(value) and Path(value).exists())].copy()
    if args.skip_vowels:
        rows = rows[rows.prompt_type != "vowel"]
    if args.limit is not None:
        rows = rows.head(args.limit)
    device = _device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForCTC.from_pretrained(
        args.model_name, local_files_only=args.local_files_only, use_safetensors=True
    ).to(device).eval()
    blank_id = int(processor.tokenizer.pad_token_id)

    audits = []
    word_rows = []
    started = time.perf_counter()
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        record = {"user_id": row.user_id, "group_name": row.group_name, "status": "error", "error": ""}
        try:
            waveform, _ = librosa.load(row.audio_path, sr=16_000, mono=True)
            trimmed, bounds = librosa.effects.trim(waveform, top_db=30)
            if len(trimmed) < 1_600:
                raise ValueError("Audio remaining after trimming is shorter than 100 ms")
            inputs = processor(trimmed, sampling_rate=16_000, return_tensors="pt")
            with torch.no_grad():
                logits = model(inputs.input_values.to(device)).logits[0]
                log_probs = torch.log_softmax(logits, dim=-1).cpu().numpy()
            target_ids = processor.tokenizer(row.ctc_text, add_special_tokens=False).input_ids
            if not target_ids:
                raise ValueError("Prompt has no CTC-compatible text")
            spans = ctc_viterbi_align(log_probs, target_ids, blank_id)
            tokens = processor.tokenizer.convert_ids_to_tokens(target_ids)
            offset_seconds = float(bounds[0] / 16_000)
            frame_seconds = float(len(trimmed) / 16_000 / len(log_probs))
            aligned_words = _word_spans(tokens, spans, frame_seconds, offset_seconds)
            expected_words = json.loads(row.word_pronunciations)
            if [item["word"].lower() for item in expected_words] != [item["word"] for item in aligned_words]:
                raise ValueError("Aligned words differ from pronunciation words")
            for word_index, (aligned, expected) in enumerate(zip(aligned_words, expected_words, strict=True)):
                word_rows.append(
                    {
                        "user_id": row.user_id,
                        "group_name": row.group_name,
                        "class_id": row.class_id,
                        "word_index": word_index,
                        **aligned,
                        "arpabet": " ".join(expected["phones"]),
                    }
                )
            decoded = processor.batch_decode(logits.argmax(dim=-1).cpu().unsqueeze(0))[0]
            record.update(
                {
                    "status": "aligned",
                    "error": "",
                    "reference": row.ctc_text,
                    "asr_hypothesis": decoded,
                    "character_error_rate": character_error_rate(row.ctc_text, decoded),
                    "mean_token_probability": float(
                        np.exp(np.mean([span.mean_log_probability for span in spans]))
                    ),
                    "audio_seconds": len(waveform) / 16_000,
                    "trimmed_seconds": len(trimmed) / 16_000,
                    "trim_start_seconds": offset_seconds,
                    "trim_end_seconds": bounds[1] / 16_000,
                    "word_count": len(aligned_words),
                }
            )
        except Exception as exc:
            record["error"] = str(exc)
        audits.append(record)
        if position == 1 or position % 25 == 0 or position == len(rows):
            elapsed = time.perf_counter() - started
            remaining = elapsed / position * (len(rows) - position)
            print(
                f"CTC_ALIGNMENT progress={position}/{len(rows)} elapsed_seconds={elapsed:.1f} "
                f"estimated_remaining_seconds={remaining:.1f}",
                flush=True,
            )

    audit = pd.DataFrame(audits)
    output = Path(args.output)
    word_output = Path(args.word_output)
    audit_output = Path(args.audit_output)
    for path in (output, word_output, audit_output):
        path.parent.mkdir(parents=True, exist_ok=True)
    word_table = pd.DataFrame(word_rows)
    word_table.to_csv(word_output, index=False)
    audit.to_csv(audit_output, index=False)
    enriched = manifest.merge(audit, on=["user_id", "group_name"], how="left", suffixes=("", "_alignment"))
    enriched.to_csv(output, index=False)
    aligned = int((audit.status == "aligned").sum()) if len(audit) else 0
    print(f"Saved {aligned}/{len(audit)} alignments to {output}; word spans to {word_output}")


if __name__ == "__main__":
    main()
