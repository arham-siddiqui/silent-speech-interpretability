#!/usr/bin/env python3
"""Infer local RVTALL folder-to-prompt cohorts from paired sentence audio."""

from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import pandas as pd
import torch

from silent_speech_interpretability.data.ctc_alignment import character_error_rate
from silent_speech_interpretability.data.prompts import ctc_text, prompt_for, sentence_variants


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_audio.csv")
    parser.add_argument("--model-name", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", default="metadata/rvtall_audio_prompt_cohorts.csv")
    parser.add_argument("--audit-output", default="reports/results/rvtall_audio_prompt_cohort_audit.csv")
    args = parser.parse_args()

    from transformers import AutoModelForCTC, AutoProcessor

    manifest = pd.read_csv(args.manifest).fillna("")
    rows = manifest[manifest.group_name.isin(["sentences7", "sentences9", "sentences10"])].copy()
    rows = rows[rows.audio_path.map(lambda value: bool(value) and Path(value).exists())]
    device = _device(args.device)
    processor = AutoProcessor.from_pretrained(args.model_name, local_files_only=args.local_files_only)
    model = AutoModelForCTC.from_pretrained(
        args.model_name, local_files_only=args.local_files_only, use_safetensors=True
    ).to(device).eval()

    decoded_rows = []
    started = time.perf_counter()
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        waveform, _ = librosa.load(row.audio_path, sr=16_000, mono=True)
        trimmed, _ = librosa.effects.trim(waveform, top_db=30)
        inputs = processor(trimmed, sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(device)).logits
        hypothesis = processor.batch_decode(logits.argmax(dim=-1).cpu())[0]
        sentence_index = int(str(row.group_name).replace("sentences", ""))
        for cohort, transcript in sentence_variants(sentence_index).items():
            reference = ctc_text(transcript)
            decoded_rows.append(
                {
                    "user_id": int(row.user_id),
                    "group_name": row.group_name,
                    "candidate_cohort": cohort,
                    "reference": reference,
                    "asr_hypothesis": hypothesis,
                    "character_error_rate": character_error_rate(reference, hypothesis),
                }
            )
        if position == 1 or position % 15 == 0 or position == len(rows):
            elapsed = time.perf_counter() - started
            remaining = elapsed / position * (len(rows) - position)
            print(
                f"COHORT_AUDIT progress={position}/{len(rows)} elapsed_seconds={elapsed:.1f} "
                f"estimated_remaining_seconds={remaining:.1f}",
                flush=True,
            )

    audit = pd.DataFrame(decoded_rows)
    scores = audit.groupby(["user_id", "candidate_cohort"], as_index=False).character_error_rate.mean()
    published_by_user = {
        int(user_id): str(prompt_for(int(user_id), "sentences7")["prompt_cohort"])
        for user_id in scores.user_id.unique()
    }
    scores["published_cohort"] = scores.user_id.map(published_by_user)
    cohort_scores = scores.groupby(["published_cohort", "candidate_cohort"]).character_error_rate.mean()
    cohorts = [item[0] for item in sentence_variants(7).items()]
    assignments = []
    for candidate_order in itertools.permutations(cohorts):
        assignment = dict(zip(cohorts, candidate_order, strict=True))
        assignment_score = sum(cohort_scores.loc[published, inferred] for published, inferred in assignment.items())
        assignments.append((float(assignment_score), assignment))
    assignments.sort(key=lambda item: item[0])
    best_assignment = assignments[0][1]
    assignment_margin = assignments[1][0] - assignments[0][0]

    mappings = []
    for user_id, user_scores in scores.groupby("user_id"):
        ranked = user_scores.sort_values("character_error_rate").reset_index(drop=True)
        published = published_by_user[int(user_id)]
        inferred = best_assignment[published]
        selected_cer = float(user_scores.loc[user_scores.candidate_cohort == inferred, "character_error_rate"].iloc[0])
        mappings.append(
            {
                "user_id": int(user_id),
                "published_cohort": published,
                "inferred_cohort": inferred,
                "speaker_best_cohort": ranked.loc[0, "candidate_cohort"],
                "mean_cer": selected_cer,
                "cer_margin_to_runner_up": ranked.loc[1, "character_error_rate"] - ranked.loc[0, "character_error_rate"],
                "matches_speaker_best": inferred == ranked.loc[0, "candidate_cohort"],
                "cohort_assignment_margin": assignment_margin,
                "matches_published_id": published == inferred,
            }
        )
    mapping = pd.DataFrame(mappings).sort_values("user_id")
    output = Path(args.output)
    audit_output = Path(args.audit_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    mapping.to_csv(output, index=False)
    audit.merge(scores, on=["user_id", "candidate_cohort"], suffixes=("", "_speaker_mean")).to_csv(
        audit_output, index=False
    )
    print(f"Saved {len(mapping)} inferred audio-folder cohorts to {output}")


if __name__ == "__main__":
    main()
