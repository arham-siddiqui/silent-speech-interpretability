#!/usr/bin/env python3
"""Train fold-specific sparse autoencoders on HuBERT-student bottlenecks."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.interp.sae import SAETrainConfig, normalized_arrays, sae_metrics, save_sae, train_sae


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
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--activations-dir", default="artifacts/activations/hubert_cv")
    parser.add_argument("--output-dir", default="artifacts/sae/hubert_bottleneck")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--results-output", default="reports/results/hubert_bottleneck_sae_results.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    sae_config = config["interpretability"]["sae"]
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    device = _device(args.device)
    rows = []
    start_all = time.perf_counter()

    for position, fold in enumerate(folds, start=1):
        data = np.load(Path(args.activations_dir) / f"fold_{fold}_activations.npz")
        train_x = data["train_bottleneck"].astype(np.float32)
        val_x = data["val_bottleneck"].astype(np.float32)
        test_x = data["test_bottleneck"].astype(np.float32)
        train_config = SAETrainConfig(
            feature_dim=int(train_x.shape[1] * sae_config.get("expansion_factor", 8)),
            l1_coeff=float(sae_config.get("l1_coeff", 1e-4)),
            top_k=int(sae_config.get("top_k", 32)),
            steps=int(args.steps or sae_config.get("steps", 5000)),
            seed=int(config["project"]["seed"]) + fold,
        )
        start = time.perf_counter()
        model, val_metrics, mean, std = train_sae(train_x, val_x, train_config, device)
        _, _, (train_normalized, _val_normalized, test_normalized) = normalized_arrays(train_x, val_x, test_x)
        train_metrics = sae_metrics(model, train_normalized, device)
        test_metrics = sae_metrics(model, test_normalized, device)
        elapsed = time.perf_counter() - start
        metrics = {"train": train_metrics, "val": val_metrics, "test": test_metrics, "elapsed_seconds": elapsed}
        save_sae(Path(args.output_dir) / f"fold_{fold}_sae.pt", model, mean, std, train_config, metrics)
        rows.append(
            {
                "fold": fold,
                "feature_dim": train_config.feature_dim,
                "l1_coeff": train_config.l1_coeff,
                "top_k": train_config.top_k,
                "steps_run": int(val_metrics["steps_run"]),
                "train_mse": train_metrics["mse"],
                "val_mse": val_metrics["mse"],
                "test_mse": test_metrics["mse"],
                "test_explained_variance": test_metrics["explained_variance"],
                "test_mean_active_features": test_metrics["mean_active_features"],
                "test_feature_density": test_metrics["feature_density"],
                "test_dead_feature_fraction": test_metrics["dead_feature_fraction"],
                "elapsed_seconds": elapsed,
            }
        )
        average = (time.perf_counter() - start_all) / position
        remaining = average * (len(folds) - position)
        print(
            f"SAE_PROGRESS fold={fold} elapsed_seconds={elapsed:.1f} val_mse={val_metrics['mse']:.4f} "
            f"test_explained_variance={test_metrics['explained_variance']:.4f} "
            f"estimated_remaining_seconds={remaining:.1f}",
            flush=True,
        )

    output = Path(args.results_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("fold").to_csv(output, index=False)
    print(f"Saved SAE results to {output}")


if __name__ == "__main__":
    main()
