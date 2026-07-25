#!/usr/bin/env python3
"""Link fold-local sparse HuBERT-student features to held-out phone trajectories."""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

from silent_speech_interpretability.interp.sae import encode_sae, load_sae


def _target_index(targets) -> dict[tuple[str, str], int]:
    return {
        (str(user), str(group)): index
        for index, (user, group) in enumerate(zip(targets["user_ids"], targets["group_names"], strict=True))
    }


def _arrays(data, split: str, targets, target_index, model, payload):
    pairs = [(str(user), str(group)) for user, group in zip(data[f"{split}_user_ids"], data[f"{split}_group_names"], strict=True)]
    selected = [(row, pair) for row, pair in enumerate(pairs) if pair in target_index]
    rows = [row for row, _pair in selected]
    target_rows = [target_index[pair] for _row, pair in selected]
    features = encode_sae(
        model,
        data[f"{split}_bottleneck"][rows].astype(np.float32),
        np.asarray(payload["input_mean"]),
        np.asarray(payload["input_std"]),
        torch.device("cpu"),
    )
    return features, targets["values"][target_rows].astype(np.float32), targets["labels"][target_rows].astype(np.int64)


def _class_means(y: np.ndarray, labels: np.ndarray) -> dict[int, np.ndarray]:
    return {int(label): y[labels == label].mean(axis=0) for label in np.unique(labels)}


def _baseline(means: dict[int, np.ndarray], labels: np.ndarray) -> np.ndarray:
    return np.stack([means[int(label)] for label in labels])


def _feature_r2(y: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    return np.asarray([r2_score(y[:, :, index].reshape(-1), prediction[:, :, index].reshape(-1)) for index in range(y.shape[2])])


def _predict(scaler: StandardScaler, model: Ridge, x: np.ndarray, base: np.ndarray, zero_features=None) -> np.ndarray:
    standardized = scaler.transform(x.astype(np.float64, copy=False))
    if zero_features is not None:
        standardized[:, zero_features] = 0.0
    # Accelerate can emit overflow warnings for finite macOS matmuls; the explicit
    # finite check retains the real safety condition.
    with warnings.catch_warnings(), np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        residual = (standardized @ model.coef_.T).reshape(base.shape)
    if not np.isfinite(residual).all():
        raise ValueError("Non-finite sparse phone-probe prediction")
    return base + residual


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default="artifacts/forced_alignment/phone_ctc_sentence_targets.npz")
    parser.add_argument("--activations-dir", default="artifacts/activations/hubert_temporal4_cv")
    parser.add_argument("--sae-dir", default="artifacts/sae/hubert_temporal4_bottleneck")
    parser.add_argument("--content-rankings", default="reports/results/hubert_temporal_bottleneck_feature_rankings.csv")
    parser.add_argument("--ks", default="10,25,50")
    parser.add_argument("--random-draws", type=int, default=100)
    parser.add_argument("--results-output", default="reports/results/phone_sparse_feature_ablation.csv")
    parser.add_argument("--rankings-output", default="reports/results/phone_sparse_feature_rankings.csv")
    args = parser.parse_args()

    targets = np.load(args.targets)
    target_index = _target_index(targets)
    feature_names = targets["feature_names"].astype(str)
    content = pd.read_csv(args.content_rankings)
    ks = [int(value) for value in args.ks.split(",") if value.strip()]
    result_rows, ranking_rows = [], []
    for fold in range(5):
        rng = np.random.default_rng(10_000 + fold)
        data = np.load(Path(args.activations_dir) / f"fold_{fold}_activations.npz")
        sae, payload = load_sae(Path(args.sae_dir) / f"fold_{fold}_sae.pt", "cpu")
        train = _arrays(data, "train", targets, target_index, sae, payload)
        val = _arrays(data, "val", targets, target_index, sae, payload)
        test = _arrays(data, "test", targets, target_index, sae, payload)
        train_means = _class_means(train[1], train[2])
        train_base, val_base = _baseline(train_means, train[2]), _baseline(train_means, val[2])
        best_score, best_alpha = -np.inf, 1.0
        for alpha in (1.0, 10.0, 100.0, 1000.0, 10_000.0):
            scaler = StandardScaler().fit(train[0].astype(np.float64))
            probe = Ridge(alpha=alpha, solver="lsqr", tol=1e-6, fit_intercept=False).fit(
                scaler.transform(train[0].astype(np.float64)),
                (train[1] - train_base).reshape(len(train[1]), -1).astype(np.float64)
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                prediction = _predict(scaler, probe, val[0], val_base)
            score = float(np.mean(_feature_r2(val[1], prediction)))
            if score > best_score:
                best_score, best_alpha = score, alpha

        combined_x = np.concatenate([train[0], val[0]])
        combined_y = np.concatenate([train[1], val[1]])
        combined_labels = np.concatenate([train[2], val[2]])
        means = _class_means(combined_y, combined_labels)
        combined_base = _baseline(means, combined_labels)
        test_base = _baseline(means, test[2])
        scaler = StandardScaler().fit(combined_x.astype(np.float64))
        probe = Ridge(alpha=best_alpha, solver="lsqr", tol=1e-6, fit_intercept=False).fit(
            scaler.transform(combined_x.astype(np.float64)),
            (combined_y - combined_base).reshape(len(combined_y), -1).astype(np.float64)
        )
        full_prediction = _predict(scaler, probe, test[0], test_base)
        full_r2 = _feature_r2(test[1], full_prediction)
        base_r2 = _feature_r2(test[1], test_base)
        coefficient = np.abs(probe.coef_).reshape(4, len(feature_names), -1)
        coefficient_norm = np.linalg.norm(coefficient, axis=(0, 1))
        phone_order = np.argsort(-coefficient_norm)
        fold_content = content[content.fold == fold].set_index("feature")
        content_rank = fold_content["rank"].to_dict()
        for rank, feature in enumerate(phone_order):
            ranking_rows.append(
                {"fold": fold, "feature": int(feature), "phone_rank": rank,
                 "phone_coefficient_norm": coefficient_norm[feature],
                 "content_rank": int(content_rank[int(feature)]),
                 "content_score": float(fold_content.loc[int(feature), "content_score"]),
                 "activation_frequency": float(fold_content.loc[int(feature), "activation_frequency"])}
            )

        result_rows.append(
            {"fold": fold, "mode": "full", "k": 0, "macro_r2": float(full_r2.mean()),
             "baseline_macro_r2": float(base_r2.mean()), "delta_r2": float((full_r2 - base_r2).mean()),
             "ablation_loss": 0.0, "random_mean_loss": 0.0, "random_p_value": 1.0,
             "content_top_k_overlap": 0, "alpha": best_alpha, "num_test": len(test[1])}
        )
        for k in ks:
            top = phone_order[:k]
            top_prediction = _predict(scaler, probe, test[0], test_base, top)
            top_macro = float(_feature_r2(test[1], top_prediction).mean())
            top_loss = float(full_r2.mean() - top_macro)
            random_losses = []
            for _draw in range(args.random_draws):
                random_features = rng.choice(len(phone_order), size=k, replace=False)
                random_prediction = _predict(scaler, probe, test[0], test_base, random_features)
                random_losses.append(float(full_r2.mean() - _feature_r2(test[1], random_prediction).mean()))
            content_top = set(fold_content.nsmallest(k, "rank").index.astype(int))
            overlap = len(set(top.astype(int)) & content_top)
            result_rows.append(
                {"fold": fold, "mode": "phone_top_ablation", "k": k, "macro_r2": top_macro,
                 "baseline_macro_r2": float(base_r2.mean()), "delta_r2": top_macro - float(base_r2.mean()),
                 "ablation_loss": top_loss, "random_mean_loss": float(np.mean(random_losses)),
                 "random_p_value": float((1 + np.sum(np.asarray(random_losses) >= top_loss)) / (1 + len(random_losses))),
                 "content_top_k_overlap": overlap, "alpha": best_alpha, "num_test": len(test[1])}
            )
        print(f"PHONE_SPARSE fold={fold} full_delta_r2={(full_r2-base_r2).mean():+.3f}", flush=True)

    results_output, rankings_output = Path(args.results_output), Path(args.rankings_output)
    results_output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result_rows).to_csv(results_output, index=False)
    pd.DataFrame(ranking_rows).to_csv(rankings_output, index=False)
    print(f"Saved sparse phone linkage results to {results_output}")


if __name__ == "__main__":
    main()
