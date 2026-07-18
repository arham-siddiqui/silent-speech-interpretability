#!/usr/bin/env python3
"""Extract ordered relative-time HuBERT segment targets from aligned audio."""

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
    parser.add_argument("--manifest", default="artifacts/manifest_with_audio.csv")
    parser.add_argument("--model-name", default="facebook/hubert-base-ls960")
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="artifacts/teacher_targets/facebook_hubert-base-ls960_temporal4_targets.npz")
    parser.add_argument("--audit-output", default="reports/results/hubert_temporal_target_audit.csv")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    rows = manifest.copy()
    rows["audio_path"] = rows["audio_path"].fillna("").astype(str)
    rows = rows[rows["audio_path"].map(lambda value: bool(value) and Path(value).exists())]
    rows = rows.drop_duplicates(["user_id", "group_name"]).sort_values(["user_id", "group_name"])
    if args.limit is not None:
        rows = rows.head(args.limit)
    if rows.empty:
        raise RuntimeError("No existing audio paths in the manifest")

    teacher = SSLTeacher(args.model_name, device=args.device, local_files_only=args.local_files_only)
    targets = []
    audit_rows = []
    started = time.perf_counter()
    for position, row in enumerate(rows.itertuples(index=False), start=1):
        extracted = teacher.extract_hidden_states(row.audio_path, trim_silence=True)
        temporal = relative_segment_pool(extracted["hidden_states"], args.segments)
        targets.append(temporal.reshape(-1))
        audit_rows.append(
            {
                "user_id": str(row.user_id),
                "group_name": str(row.group_name),
                "class_id": int(row.class_id),
                "hidden_frames": int(len(extracted["hidden_states"])),
                "original_seconds": float(extracted["original_samples"] / extracted["sample_rate"]),
                "used_seconds": float(extracted["used_samples"] / extracted["sample_rate"]),
            }
        )
        if position == 1 or position % 50 == 0 or position == len(rows):
            elapsed = time.perf_counter() - started
            remaining = elapsed / position * (len(rows) - position)
            print(
                f"TEMPORAL_EXTRACTION progress={position}/{len(rows)} "
                f"elapsed_seconds={elapsed:.1f} estimated_remaining_seconds={remaining:.1f}",
                flush=True,
            )

    target_array = np.stack(targets).astype(np.float32)
    hidden_dim = target_array.shape[1] // args.segments
    save_teacher_targets(
        args.output,
        target_array,
        rows["class_id"].to_numpy(dtype=np.int64),
        rows["user_id"].astype(str).to_numpy(),
        rows["group_name"].astype(str).to_numpy(),
        target_name=f"{args.model_name}:trimmed-relative-{args.segments}-segment",
        target_shape=(args.segments, hidden_dim),
    )
    audit = pd.DataFrame(audit_rows)
    audit["segments"] = args.segments
    audit["segment_dim"] = hidden_dim
    audit["elapsed_seconds"] = time.perf_counter() - started
    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(audit_path, index=False)
    print(f"Saved {len(target_array)} temporal targets with shape {args.segments}x{hidden_dim} to {args.output}")


if __name__ == "__main__":
    main()
