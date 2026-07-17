#!/usr/bin/env python3
"""Prepare teacher target NPZ files for student distillation.

This first implementation supports deterministic synthetic teacher targets from an
existing embedding NPZ. Real audio-teacher extraction can write the same NPZ schema.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.models.teachers.teacher_targets import make_class_structured_targets, save_teacher_targets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--source-npz", default=None, help="Embedding NPZ used for labels/user/group keys.")
    parser.add_argument("--output", default=None)
    parser.add_argument("--target-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--target-name", default="synthetic_audio_teacher")
    args = parser.parse_args()

    config = load_config(args.config)
    source = Path(args.source_npz or "artifacts/embeddings/speaker_cv/fold_0/lip_embeddings.npz")
    output = Path(args.output or Path(config["data"]["teacher_targets_dir"]) / "synthetic_audio_teacher_targets.npz")
    with np.load(source, allow_pickle=True) as data:
        labels = data["labels"].astype(np.int64)
        user_ids = (data["user_ids"] if "user_ids" in data.files else data["users"]).astype(str)
        group_names = (data["group_names"] if "group_names" in data.files else data["label_names"]).astype(str)
    rows = pd.DataFrame({"user_id": user_ids, "group_name": group_names, "label": labels})
    grouped = rows.groupby(["user_id", "group_name"], sort=True)["label"].agg(lambda values: int(values.iloc[0])).reset_index()
    targets = make_class_structured_targets(grouped["label"].to_numpy(), target_dim=args.target_dim, seed=args.seed, noise=args.noise)
    save_teacher_targets(
        output,
        targets,
        grouped["label"].to_numpy(),
        grouped["user_id"].to_numpy(),
        grouped["group_name"].to_numpy(),
        target_name=args.target_name,
    )
    print(f"Saved {len(targets)} teacher targets to {output}")


if __name__ == "__main__":
    main()
