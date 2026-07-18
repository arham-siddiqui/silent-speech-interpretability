#!/usr/bin/env python3
"""Train a segment-level silent-sensor student against temporal HuBERT targets."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.models.students.temporal_sensor_student import TemporalSensorStudent
from silent_speech_interpretability.models.teachers.teacher_targets import load_teacher_targets


def _index(data, prefix: str) -> dict[tuple[str, str], int]:
    return {
        (str(user), str(group)): index
        for index, (user, group) in enumerate(zip(data[f"{prefix}_user_ids"], data[f"{prefix}_group_names"], strict=True))
    }


def _arrays(data, teacher: dict[str, object], modalities: list[str], speakers: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
    indices = {modality: _index(data, modality) for modality in modalities}
    speaker_set = {str(speaker) for speaker in speakers}
    pairs = set(teacher["pairs"])
    for modality in modalities:
        pairs &= set(indices[modality])
    pairs = sorted(pair for pair in pairs if pair[0] in speaker_set)
    x = np.concatenate(
        [data[f"{modality}_values"][[indices[modality][pair] for pair in pairs]] for modality in modalities],
        axis=2,
    ).astype(np.float32)
    target_indices = [teacher["index"][pair] for pair in pairs]
    segments, segment_dim = teacher["target_shape"]
    targets = teacher["targets"][target_indices].reshape(len(pairs), segments, segment_dim).astype(np.float32)
    labels = teacher["labels"][target_indices].astype(np.int64)
    return x, targets, labels, pairs


def _normalize_targets(targets: torch.Tensor) -> torch.Tensor:
    return F.normalize(targets, p=2, dim=-1)


def _metrics(model, x: np.ndarray, targets: np.ndarray, labels: np.ndarray, device: torch.device) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(x, dtype=torch.float32, device=device))
    predicted = output["target"]
    target = _normalize_targets(torch.tensor(targets, dtype=torch.float32, device=device))
    true_cosine = F.cosine_similarity(predicted, target, dim=-1)
    reversed_cosine = F.cosine_similarity(predicted, target.flip(1), dim=-1)
    shifted_cosine = F.cosine_similarity(predicted, target.roll(1, dims=1), dim=-1)
    return {
        "accuracy": float((output["logits"].argmax(dim=1).cpu().numpy() == labels).mean()),
        "segment_cosine": float(true_cosine.mean().item()),
        "reversed_segment_cosine": float(reversed_cosine.mean().item()),
        "shifted_segment_cosine": float(shifted_cosine.mean().item()),
        "order_margin_reversed": float((true_cosine - reversed_cosine).mean().item()),
        "target_mse": float(((predicted - target) ** 2).sum(dim=-1).mean().item()),
    }


def _train(
    model: TemporalSensorStudent,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    max_epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[TemporalSensorStudent, int]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(train[0], dtype=torch.float32),
        torch.tensor(train[1], dtype=torch.float32),
        torch.tensor(train[2], dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model.to(device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    patience = 0
    best_epoch = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, targets, labels in loader:
            x, targets, labels = x.to(device), targets.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(x)
            alignment = ((output["target"] - _normalize_targets(targets)) ** 2).sum(dim=-1).mean()
            loss = alignment + 0.2 * F.cross_entropy(output["logits"], labels)
            loss.backward()
            optimizer.step()
        val_metrics = _metrics(model, *val, device)
        val_loss = val_metrics["target_mse"] + 0.2 * (1.0 - val_metrics["accuracy"])
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            patience = 0
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def _write_report(path: Path, results: pd.DataFrame, fixed: pd.DataFrame, modalities: list[str]) -> None:
    fixed = fixed.set_index("fold")
    rows = "\n".join(
        f"| {int(row.fold)} | {100*row.accuracy:.1f}% | {100*fixed.loc[int(row.fold), 'accuracy']:.1f}% | "
        f"{row.segment_cosine:.3f} | {fixed.loc[int(row.fold), 'segment_cosine']:.3f} | "
        f"{row.reversed_segment_cosine:.3f} | {row.order_margin_reversed:+.3f} |"
        for row in results.itertuples(index=False)
    )
    report = f"""# Temporal Silent-Sensor To HuBERT Alignment

Fold-specific lip, laser, mmWave, and UWB encoders now expose four relative-time
activation segments. A shared segment student maps their concatenated temporal
representations to the four-segment HuBERT teacher.

## Setup

- Modalities: {", ".join(modalities)}
- Silent input: four 128-D segments per modality, averaged across repetitions
- Teacher: four silence-trimmed 768-D HuBERT segments
- Evaluation: five speaker-disjoint, encoder-disjoint folds
- Controls: reversed and shifted teacher segment order

## Results

| Fold | Temporal-Sensor Accuracy | Fixed-Embedding Accuracy | Sensor Segment Cosine | Fixed Segment Cosine | Reversed Cosine | Order Margin |
|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Aggregate

- Temporal-sensor class accuracy: **{100*results.accuracy.mean():.1f}% +/- {100*results.accuracy.std(ddof=1):.1f}%**.
- Fixed-embedding temporal-student accuracy: **{100*fixed.accuracy.mean():.1f}%**.
- Temporal-sensor true-order cosine: **{results.segment_cosine.mean():.3f}**.
- Fixed-embedding true-order cosine: **{fixed.segment_cosine.mean():.3f}**.
- Temporal-sensor reversed-order cosine: **{results.reversed_segment_cosine.mean():.3f}**.
- Temporal-sensor true-versus-reversed margin: **{results.order_margin_reversed.mean():+.3f}**.

This tests whether temporal silent-sensor states contain ordered HuBERT information. It
does not imply frame-exact synchronization because each modality is pooled into relative
regions and repetitions are averaged within speaker/utterance pairs.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--activations-dir", default="artifacts/activations/temporal_sensors")
    parser.add_argument("--teacher-targets", default="artifacts/teacher_targets/facebook_hubert-base-ls960_temporal4_targets.npz")
    parser.add_argument("--modalities", default="lip,laser,mmwave,uwb")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="artifacts/students/temporal_sensor_cv")
    parser.add_argument("--output", default="reports/results/temporal_sensor_student_cv.csv")
    parser.add_argument("--fixed-results", default="reports/results/hubert_temporal_teacher_student_cv.csv")
    parser.add_argument("--report-output", default="reports/temporal_sensor_hubert_alignment.md")
    args = parser.parse_args()

    config = load_config(args.config)
    teacher = load_teacher_targets(args.teacher_targets)
    modalities = [value.strip() for value in args.modalities.split(",") if value.strip()]
    folds_requested = {int(value) for value in args.folds.split(",") if value.strip()}
    seed_paths, _ = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = [fold for fold in make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"]) if int(fold["fold"]) in folds_requested]
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold"])
        data = np.load(Path(args.activations_dir) / f"fold_{fold_id}_temporal_sensors.npz")
        train = _arrays(data, teacher, modalities, fold["train_speakers"])
        val = _arrays(data, teacher, modalities, fold["val_speakers"])
        test = _arrays(data, teacher, modalities, fold["test_speakers"])
        input_mean = train[0].mean(axis=(0, 1), keepdims=True)
        input_std = train[0].std(axis=(0, 1), keepdims=True) + 1e-6
        teacher_center = train[1].mean(axis=0, keepdims=True)
        prepared = []
        for x, targets, labels, _pairs in (train, val, test):
            prepared.append(((x - input_mean) / input_std, targets - teacher_center, labels))
        seed = int(config["project"]["seed"]) + fold_id
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = TemporalSensorStudent(
            input_dim=prepared[0][0].shape[2],
            target_dim=int(teacher["target_shape"][1]),
            hidden_dim=int(config["student"]["hidden_dim"]),
            bottleneck_dim=int(config["student"]["bottleneck_dim"]),
            num_classes=int(config["classes"]["num_classes"]),
            num_segments=int(teacher["target_shape"][0]),
        )
        model, best_epoch = _train(model, prepared[0], prepared[1], device, args.max_epochs, args.batch_size, seed)
        metrics = _metrics(model, *prepared[2], device)
        rows.append({"fold": fold_id, "best_epoch": best_epoch, "num_train": len(train[3]), "num_val": len(val[3]), "num_test": len(test[3]), **metrics})
        torch.save(
            {
                "state_dict": model.state_dict(),
                "modalities": modalities,
                "input_mean": input_mean,
                "input_std": input_std,
                "teacher_center": teacher_center,
                "input_dim": prepared[0][0].shape[2],
                "target_dim": int(teacher["target_shape"][1]),
                "num_segments": int(teacher["target_shape"][0]),
            },
            output_dir / f"fold_{fold_id}_temporal_sensor_student.pt",
        )
        elapsed = time.perf_counter() - started
        remaining = elapsed / position * (len(folds) - position)
        print(f"TEMPORAL_SENSOR_CV fold={fold_id} cosine={metrics['segment_cosine']:.3f} estimated_remaining_seconds={remaining:.1f}", flush=True)

    results = pd.DataFrame(rows).sort_values("fold")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    _write_report(Path(args.report_output), results, pd.read_csv(args.fixed_results), modalities)
    print(f"Saved temporal sensor CV report to {args.report_output}")


if __name__ == "__main__":
    main()
