#!/usr/bin/env python3
"""Extract ordered HuBERT targets for every external radar/audio sample."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.models.teachers.ssl_teacher import SSLTeacher, relative_segment_pool
from silent_speech_interpretability.models.teachers.teacher_targets import save_teacher_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts/external/radar_command_words/manifest.csv",
    )
    parser.add_argument("--model-name", default="facebook/hubert-base-ls960")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--output",
        default="artifacts/external/radar_command_words/hubert_temporal4_targets.npz",
    )
    parser.add_argument(
        "--audit-output",
        default="artifacts/external/radar_command_words/hubert_target_audit.csv",
    )
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).sort_values(
        ["user_id", "session_id", "class_id", "repetition"], ignore_index=True
    )
    missing_audio = manifest[~manifest["audio_path"].map(lambda value: Path(str(value)).is_file())]
    if not missing_audio.empty:
        raise FileNotFoundError(f"{len(missing_audio)} manifest audio paths do not exist")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("External manifest contains duplicate sample identifiers")
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    if manifest.empty:
        raise RuntimeError("External corpus manifest is empty")

    teacher = SSLTeacher(args.model_name, device=args.device, local_files_only=args.local_files_only)
    targets = []
    audits = []
    started = time.perf_counter()
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        extracted = teacher.extract_hidden_states(row.audio_path, trim_silence=True)
        pooled = relative_segment_pool(extracted["hidden_states"], args.segments)
        targets.append(pooled.reshape(-1))
        audits.append(
            {
                "sample_id": row.sample_id,
                "user_id": row.user_id,
                "session_id": row.session_id,
                "class_id": int(row.class_id),
                "hidden_frames": int(len(extracted["hidden_states"])),
                "original_seconds": float(extracted["original_samples"] / extracted["sample_rate"]),
                "used_seconds": float(extracted["used_samples"] / extracted["sample_rate"]),
            }
        )
        if position == 1 or position % 50 == 0 or position == len(manifest):
            elapsed = time.perf_counter() - started
            remaining = elapsed / position * (len(manifest) - position)
            print(
                f"EXTERNAL_HUBERT progress={position}/{len(manifest)} "
                f"elapsed_seconds={elapsed:.1f} estimated_remaining_seconds={remaining:.1f}",
                flush=True,
            )

    target_array = np.stack(targets).astype(np.float32)
    hidden_dim = target_array.shape[1] // args.segments
    save_teacher_targets(
        args.output,
        target_array,
        manifest["class_id"].to_numpy(dtype=np.int64),
        manifest["user_id"].astype(str).to_numpy(),
        manifest["sample_id"].astype(str).to_numpy(),
        target_name=f"{args.model_name}:external-trimmed-relative-{args.segments}-segment",
        target_shape=(args.segments, hidden_dim),
    )
    audit = pd.DataFrame(audits)
    audit["segments"] = args.segments
    audit["segment_dim"] = hidden_dim
    audit["total_elapsed_seconds"] = time.perf_counter() - started
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print(f"Saved {len(target_array)} external HuBERT targets to {args.output}")


if __name__ == "__main__":
    main()
