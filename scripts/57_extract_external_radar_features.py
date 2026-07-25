#!/usr/bin/env python3
"""Extract fixed-shape temporal features from the external radar corpus."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from silent_speech_interpretability.data.external_radar import (
    radar_segment_features,
    read_radar_binary,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="artifacts/external/radar_command_words/manifest.csv",
    )
    parser.add_argument(
        "--output",
        default="artifacts/external/radar_command_words/radar_temporal4_features.npz",
    )
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--frequency-bands", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if args.limit is not None:
        manifest = manifest.head(args.limit)
    if manifest.empty:
        raise RuntimeError("External radar manifest is empty")

    features = []
    started = time.perf_counter()
    for position, row in enumerate(manifest.itertuples(index=False), start=1):
        recording = read_radar_binary(row.radar_path)
        features.append(
            radar_segment_features(
                recording,
                num_segments=args.segments,
                num_frequency_bands=args.frequency_bands,
            )
        )
        if position == 1 or position % 100 == 0 or position == len(manifest):
            elapsed = time.perf_counter() - started
            remaining = elapsed / position * (len(manifest) - position)
            print(
                f"RADAR_FEATURES progress={position}/{len(manifest)} "
                f"elapsed_seconds={elapsed:.1f} estimated_remaining_seconds={remaining:.1f}",
                flush=True,
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        features=np.stack(features).astype(np.float32),
        sample_ids=manifest["sample_id"].to_numpy(dtype=str),
        class_ids=manifest["class_id"].to_numpy(dtype=np.int64),
        user_ids=manifest["user_id"].to_numpy(dtype=str),
        session_ids=manifest["session_id"].to_numpy(dtype=str),
        labels=manifest["label"].to_numpy(dtype=str),
        segments=np.asarray(args.segments, dtype=np.int64),
        frequency_bands=np.asarray(args.frequency_bands, dtype=np.int64),
        spectrum_indices=np.asarray([1, 7], dtype=np.int64),
    )
    print(f"Saved radar features with shape {np.stack(features).shape} to {output}")


if __name__ == "__main__":
    main()
