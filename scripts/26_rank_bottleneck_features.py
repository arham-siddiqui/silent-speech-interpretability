#!/usr/bin/env python3
"""Rank sparse bottleneck features and estimate cross-fold stability."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from silent_speech_interpretability.interp.feature_ranking import decoder_stability, feature_rankings
from silent_speech_interpretability.interp.sae import encode_sae, load_sae


def _best_group(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    groups = np.unique(labels)
    means = np.stack([features[labels == group].mean(axis=0) for group in groups])
    indices = means.argmax(axis=0)
    return groups[indices], means[indices, np.arange(features.shape[1])]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--activations-dir", default="artifacts/activations/hubert_cv")
    parser.add_argument("--sae-dir", default="artifacts/sae/hubert_bottleneck")
    parser.add_argument("--output", default="reports/results/hubert_bottleneck_feature_rankings.csv")
    args = parser.parse_args()

    device = torch.device("cpu")
    decoder_matrices = {}
    loaded = {}
    for fold in range(5):
        model, payload = load_sae(Path(args.sae_dir) / f"fold_{fold}_sae.pt", device)
        decoder_matrices[fold] = model.decoder.weight.detach().cpu().numpy().astype(np.float32)
        loaded[fold] = (model, payload)

    rows = []
    for fold in range(5):
        model, payload = loaded[fold]
        data = np.load(Path(args.activations_dir) / f"fold_{fold}_activations.npz")
        features = encode_sae(
            model,
            data["train_bottleneck"].astype(np.float32),
            np.asarray(payload["input_mean"]),
            np.asarray(payload["input_std"]),
            device,
        )
        rankings = feature_rankings(
            features,
            data["train_labels"],
            data["train_type_labels"],
            data["train_user_ids"],
        )
        best_class, best_class_mean = _best_group(features, data["train_labels"])
        best_type, best_type_mean = _best_group(features, data["train_type_labels"])
        rank_position = np.empty(model.feature_dim, dtype=np.int64)
        rank_position[rankings["rank"]] = np.arange(model.feature_dim)
        other_decoders = [matrix for other_fold, matrix in decoder_matrices.items() if other_fold != fold]
        for feature in range(model.feature_dim):
            rows.append(
                {
                    "fold": fold,
                    "feature": feature,
                    "rank": int(rank_position[feature]),
                    "valid": bool(rankings["valid"][feature]),
                    "content_score": float(rankings["content_score"][feature]),
                    "class_selectivity": float(rankings["class_selectivity"][feature]),
                    "type_selectivity": float(rankings["type_selectivity"][feature]),
                    "speaker_selectivity": float(rankings["speaker_selectivity"][feature]),
                    "activation_frequency": float(rankings["activation_frequency"][feature]),
                    "frequency_weight": float(rankings["frequency_weight"][feature]),
                    "mean_activation": float(rankings["mean_activation"][feature]),
                    "best_class": int(best_class[feature]),
                    "best_class_mean": float(best_class_mean[feature]),
                    "best_type": int(best_type[feature]),
                    "best_type_mean": float(best_type_mean[feature]),
                    "decoder_stability": decoder_stability(decoder_matrices[fold][:, feature], other_decoders),
                }
            )
        print(f"RANK_PROGRESS fold={fold} valid_features={int(rankings['valid'].sum())}", flush=True)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Saved feature rankings to {output}")


if __name__ == "__main__":
    main()
