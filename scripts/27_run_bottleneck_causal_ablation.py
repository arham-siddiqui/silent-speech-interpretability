#!/usr/bin/env python3
"""Ablate ranked sparse bottleneck features with random controls."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.interp.causal_ablation import evaluate_bottleneck, reconstructed_bottleneck
from silent_speech_interpretability.interp.sae import encode_sae, load_sae
from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent


def _student(config: dict, checkpoint_path: Path) -> ArticulatoryStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    modalities = list(checkpoint["modalities"])
    target_dim = int(checkpoint["state_dict"]["target_head.weight"].shape[0])
    model = ArticulatoryStudent(
        modalities,
        embedding_dim=int(config["modalities"][modalities[0]]["embedding_dim"]),
        hidden_dim=int(config["student"]["hidden_dim"]),
        bottleneck_dim=int(config["student"]["bottleneck_dim"]),
        target_dim=target_dim,
        num_classes=int(config["classes"]["num_classes"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--activations-dir", default="artifacts/activations/hubert_cv")
    parser.add_argument("--student-dir", default="artifacts/students/hubert_cv")
    parser.add_argument("--sae-dir", default="artifacts/sae/hubert_bottleneck")
    parser.add_argument("--rankings", default="reports/results/hubert_bottleneck_feature_rankings.csv")
    parser.add_argument("--probe-results", default="reports/results/hubert_student_probe_results.csv")
    parser.add_argument("--random-draws", type=int, default=20)
    parser.add_argument("--output", default="reports/results/hubert_bottleneck_causal_ablation.csv")
    args = parser.parse_args()

    config = load_config(args.config)
    rankings = pd.read_csv(args.rankings)
    probe_results = pd.read_csv(args.probe_results)
    rows = []
    k_values = [1, 5, 10, 20, 50]

    for fold in range(5):
        rng = np.random.default_rng(int(config["project"]["seed"]) + fold)
        data = np.load(Path(args.activations_dir) / f"fold_{fold}_activations.npz")
        student = _student(config, Path(args.student_dir) / f"fold_{fold}_teacher_student.pt")
        sae, payload = load_sae(Path(args.sae_dir) / f"fold_{fold}_sae.pt")
        mean = np.asarray(payload["input_mean"])
        std = np.asarray(payload["input_std"])
        train_features = encode_sae(sae, data["train_bottleneck"], mean, std, torch.device("cpu"))
        feature_means = train_features.mean(axis=0)

        c_value = float(
            probe_results[
                (probe_results["fold"] == fold)
                & (probe_results["task"] == "utterance_type")
                & (probe_results["layer"] == "bottleneck")
            ]["best_c"].iloc[0]
        )
        type_probe = make_pipeline(StandardScaler(), LogisticRegression(C=c_value, max_iter=2000, random_state=42 + fold))
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            type_probe.fit(
                np.concatenate([data["train_bottleneck"], data["val_bottleneck"]]),
                np.concatenate([data["train_type_labels"], data["val_type_labels"]]),
            )

        def metrics_for(values: np.ndarray) -> dict[str, float]:
            metrics = evaluate_bottleneck(
                student,
                values,
                data["test_labels"],
                data["test_teacher_hubert"],
            )
            with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                metrics["type_accuracy"] = float(np.mean(type_probe.predict(values) == data["test_type_labels"]))
            return metrics

        original = metrics_for(data["test_bottleneck"])
        reconstructed, _ = reconstructed_bottleneck(sae, data["test_bottleneck"], mean, std)
        reconstruction = metrics_for(reconstructed)
        for name, metrics in (("original", original), ("reconstruction", reconstruction)):
            rows.append({"fold": fold, "selection": name, "mode": "none", "k": 0, "draw": 0, **metrics})

        fold_rankings = rankings[(rankings["fold"] == fold) & rankings["valid"]].sort_values("rank")
        ranked_features = fold_rankings["feature"].to_numpy(dtype=int)
        valid_features = fold_rankings["feature"].to_numpy(dtype=int)
        for mode in ("zero", "mean"):
            replacement = 0.0 if mode == "zero" else feature_means
            for k in k_values:
                top_indices = ranked_features[:k]
                ablated, _ = reconstructed_bottleneck(
                    sae, data["test_bottleneck"], mean, std, top_indices, replacement
                )
                rows.append(
                    {"fold": fold, "selection": "top", "mode": mode, "k": k, "draw": 0, **metrics_for(ablated)}
                )
                for draw in range(args.random_draws):
                    random_indices = rng.choice(valid_features, size=min(k, len(valid_features)), replace=False)
                    ablated, _ = reconstructed_bottleneck(
                        sae, data["test_bottleneck"], mean, std, random_indices, replacement
                    )
                    rows.append(
                        {"fold": fold, "selection": "random", "mode": mode, "k": k, "draw": draw, **metrics_for(ablated)}
                    )
        print(f"ABLATION_PROGRESS fold={fold} reconstruction_accuracy={reconstruction['accuracy']:.4f}", flush=True)

    results = pd.DataFrame(rows)
    reconstruction = results[results["selection"] == "reconstruction"].set_index("fold")
    for metric in ("accuracy", "type_accuracy", "target_cosine", "target_mse"):
        results[f"delta_{metric}"] = results.apply(
            lambda row: float(row[metric] - reconstruction.loc[int(row["fold"]), metric]), axis=1
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    print(f"Saved causal ablations to {output}")


if __name__ == "__main__":
    main()
