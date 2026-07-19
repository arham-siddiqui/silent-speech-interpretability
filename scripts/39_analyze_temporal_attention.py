#!/usr/bin/env python3
"""Audit held-out temporal and modality weights from the attention student."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.models.students.temporal_sensor_student import ModalityAttentionTemporalStudent


def _index(data, modality: str) -> dict[tuple[str, str], int]:
    return {
        (str(user), str(group)): index
        for index, (user, group) in enumerate(
            zip(data[f"{modality}_user_ids"], data[f"{modality}_group_names"], strict=True)
        )
    }


def _test_inputs(data, modalities: list[str], speakers: list[int]) -> tuple[np.ndarray, list[tuple[str, str]]]:
    indices = {modality: _index(data, modality) for modality in modalities}
    pairs = set(indices[modalities[0]])
    for modality in modalities[1:]:
        pairs &= set(indices[modality])
    speaker_set = {str(speaker) for speaker in speakers}
    selected = sorted(pair for pair in pairs if pair[0] in speaker_set)
    values = np.concatenate(
        [data[f"{modality}_values"][[indices[modality][pair] for pair in selected]] for modality in modalities],
        axis=2,
    ).astype(np.float32)
    return values, selected


def _normalized_entropy(values: np.ndarray, axis: int) -> np.ndarray:
    values = np.clip(values, 1e-8, 1.0)
    return -(values * np.log(values)).sum(axis=axis) / np.log(values.shape[axis])


def _write_report(path: Path, summary: pd.DataFrame, modalities: list[str]) -> None:
    temporal = summary[summary["weight_type"] == "temporal_attention"]
    fusion = summary[summary["weight_type"] == "modality_fusion"]
    temporal_rows = "\n".join(
        f"| {modality.title()} | "
        + " | ".join(
            f"{float(temporal[(temporal.modality == modality) & (temporal.segment == segment)].weight_mean.iloc[0]):.3f}"
            for segment in range(4)
        )
        + " |"
        for modality in modalities
    )
    fusion_rows = "\n".join(
        f"| S{segment + 1} | "
        + " | ".join(
            f"{float(fusion[(fusion.modality == modality) & (fusion.segment == segment)].weight_mean.iloc[0]):.3f}"
            for modality in modalities
        )
        + " |"
        for segment in range(4)
    )
    temporal_entropy = float(summary.temporal_entropy_mean.dropna().iloc[0])
    fusion_entropy = float(summary.fusion_entropy_mean.dropna().iloc[0])
    report = f"""# Temporal Modality-Attention Audit

This audit summarizes learned weights on held-out speakers only. The attention model is
diagnostic rather than the selected model because it underperformed the simpler multitask
student on both classification and HuBERT alignment.

## Classification Temporal Attention

Each row sums to one across the four relative-time segments.

| Modality | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|
{temporal_rows}

## HuBERT Modality Fusion

Each row sums to one across modalities.

| Segment | {" | ".join(modality.title() for modality in modalities)} |
|---|{"---:|" * len(modalities)}
{fusion_rows}

## Concentration

- Normalized temporal-attention entropy: **{temporal_entropy:.3f}** (`1.0` is uniform).
- Normalized modality-fusion entropy: **{fusion_entropy:.3f}** (`1.0` is uniform).

High entropy indicates diffuse weighting rather than a sharp sensor/time selection. These
weights describe the learned attention model; they are not causal modality importance.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--activations-dir", default="artifacts/activations/temporal_sensors")
    parser.add_argument("--student-dir", default="artifacts/students/temporal_sensor_attention_cv")
    parser.add_argument("--output", default="reports/results/temporal_sensor_attention_weights.csv")
    parser.add_argument("--summary-output", default="reports/results/temporal_sensor_attention_weight_summary.csv")
    parser.add_argument("--report-output", default="reports/temporal_sensor_attention_audit.md")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_paths, _ = resolve_embedding_paths(
        config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"]
    )
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(
        manifest, config["splits"]["num_folds"], config["project"]["seed"]
    )
    rows = []
    entropy_rows = []
    modalities: list[str] = []
    for fold in folds:
        fold_id = int(fold["fold"])
        checkpoint = torch.load(
            Path(args.student_dir) / f"fold_{fold_id}_temporal_sensor_attention.pt",
            map_location="cpu",
            weights_only=False,
        )
        modalities = list(checkpoint["modalities"])
        data = np.load(Path(args.activations_dir) / f"fold_{fold_id}_temporal_sensors.npz")
        x, pairs = _test_inputs(data, modalities, fold["test_speakers"])
        normalized = (x - np.asarray(checkpoint["input_mean"])) / np.asarray(checkpoint["input_std"])
        model = ModalityAttentionTemporalStudent(
            input_dim=int(checkpoint["input_dim"]),
            target_dim=int(checkpoint["target_dim"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            bottleneck_dim=int(checkpoint["bottleneck_dim"]),
            num_classes=int(checkpoint["num_classes"]),
            num_segments=int(checkpoint["num_segments"]),
            num_modalities=int(checkpoint["num_modalities"]),
        )
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.no_grad():
            output = model(torch.tensor(normalized, dtype=torch.float32))
        temporal = output["temporal_attention"].numpy()
        fusion = output["modality_weights"].numpy()
        entropy_rows.append(
            {
                "fold": fold_id,
                "temporal_entropy": float(_normalized_entropy(temporal, axis=2).mean()),
                "fusion_entropy": float(_normalized_entropy(fusion, axis=2).mean()),
                "num_test": len(pairs),
            }
        )
        for modality_index, modality in enumerate(modalities):
            for segment in range(temporal.shape[2]):
                rows.append(
                    {
                        "fold": fold_id,
                        "weight_type": "temporal_attention",
                        "modality": modality,
                        "segment": segment,
                        "weight": float(temporal[:, modality_index, segment].mean()),
                    }
                )
                rows.append(
                    {
                        "fold": fold_id,
                        "weight_type": "modality_fusion",
                        "modality": modality,
                        "segment": segment,
                        "weight": float(fusion[:, segment, modality_index].mean()),
                    }
                )
        print(f"ATTENTION_AUDIT fold={fold_id} num_test={len(pairs)}", flush=True)

    results = pd.DataFrame(rows)
    entropies = pd.DataFrame(entropy_rows)
    summary = results.groupby(["weight_type", "modality", "segment"], as_index=False).agg(
        weight_mean=("weight", "mean"),
        weight_std=("weight", "std"),
        folds=("fold", "nunique"),
    )
    summary["temporal_entropy_mean"] = entropies.temporal_entropy.mean()
    summary["fusion_entropy_mean"] = entropies.fusion_entropy.mean()
    output = Path(args.output)
    summary_output = Path(args.summary_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    summary.to_csv(summary_output, index=False)
    _write_report(Path(args.report_output), summary, modalities)
    print(f"Saved temporal attention audit to {args.report_output}")


if __name__ == "__main__":
    main()
