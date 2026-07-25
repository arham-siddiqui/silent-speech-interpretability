#!/usr/bin/env python3
"""Run session-held-out radar-to-HuBERT student replication."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from silent_speech_interpretability.models.students.temporal_sensor_student import TemporalSensorStudent
from silent_speech_interpretability.models.teachers.teacher_targets import load_teacher_targets


def _normalize_targets(targets: torch.Tensor) -> torch.Tensor:
    return F.normalize(targets, p=2, dim=-1)


def _metrics(
    model: TemporalSensorStudent,
    x: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
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


def _class_prototype_cosine(
    train_targets: np.ndarray,
    train_labels: np.ndarray,
    test_targets: np.ndarray,
    test_labels: np.ndarray,
) -> float:
    prototypes = {
        label: train_targets[train_labels == label].mean(axis=0)
        for label in np.unique(train_labels)
    }
    predicted = np.stack([prototypes[label] for label in test_labels])
    predicted /= np.linalg.norm(predicted, axis=2, keepdims=True) + 1e-8
    normalized_test = test_targets / (np.linalg.norm(test_targets, axis=2, keepdims=True) + 1e-8)
    return float((predicted * normalized_test).sum(axis=2).mean())


def _train(
    model: TemporalSensorStudent,
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
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
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    model.to(device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_loss = float("inf")
    best_epoch = 0
    patience = 0
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
            best_epoch = epoch
            patience = 0
            best_state = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
        else:
            patience += 1
            if patience >= 12:
                break
    model.load_state_dict(best_state)
    return model, best_epoch


def _write_report(path: Path, results: pd.DataFrame) -> None:
    table = "\n".join(
        f"| {row.test_session} | {row.val_session} | {int(row.best_epoch)} | "
        f"{100 * row.accuracy:.1f}% | {row.segment_cosine:.3f} | "
        f"{row.class_prototype_cosine:.3f} | {row.reversed_segment_cosine:.3f} | "
        f"{row.order_margin_reversed:+.3f} | {row.residual_segment_cosine:.3f} | "
        f"{row.residual_order_margin_reversed:+.3f} |"
        for row in results.itertuples(index=False)
    )
    report = f"""# External Radar-to-HuBERT Replication

## Setup

- Corpus: Wagner et al. stepped-frequency continuous-wave radar command words
- Silent input: S12 and S32 complex radar magnitudes and temporal differences,
  pooled into four relative-time segments
- Teacher: four silence-trimmed HuBERT segments from the paired recording audio
- Evaluation: three recording-session-held-out folds across both corpus subjects
- Classes: 50 German command words; chance accuracy is 2%
- Controls: reversed teacher order, a train-only class-mean HuBERT prototype, and
  a second student trained on within-command residual HuBERT targets

Audio is used only to construct training and evaluation targets. The trained student
receives radar features alone.

## Results

| Test session | Validation session | Best epoch | Radar accuracy | Radar cosine | Class-prototype cosine | Reversed cosine | Order margin | Residual cosine | Residual order margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
{table}

## Aggregate

- Radar command accuracy: **{100 * results.accuracy.mean():.1f}%**
- Radar-to-HuBERT true-order cosine: **{results.segment_cosine.mean():.3f}**
- Label-only class-prototype cosine: **{results.class_prototype_cosine.mean():.3f}**
- Reversed-order cosine: **{results.reversed_segment_cosine.mean():.3f}**
- True-versus-reversed order margin: **{results.order_margin_reversed.mean():+.3f}**
- Within-command residual cosine: **{results.residual_segment_cosine.mean():.3f}**
- Residual true-versus-reversed margin: **{results.residual_order_margin_reversed.mean():+.3f}**

## Interpretation boundary

This is an independent cross-session replication of the project method, with a
different lab, language, command inventory, radar system, and speakers from RVTALL.
Because the external corpus contains only two subjects, it does not establish broad
speaker generalization. Positive temporal order margin supports ordered sensor-to-audio
representation alignment. The residual control is the stricter test of
exemplar-specific articulation after command identity is removed. Neither result by
itself establishes phoneme-level recovery.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--radar-features",
        default="artifacts/external/radar_command_words/radar_temporal4_features.npz",
    )
    parser.add_argument(
        "--teacher-targets",
        default="artifacts/external/radar_command_words/hubert_temporal4_targets.npz",
    )
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--bottleneck-dim", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default="artifacts/external/radar_command_words/students",
    )
    parser.add_argument(
        "--results-output",
        default="reports/tables/external_radar_student_results.csv",
    )
    parser.add_argument(
        "--report-output",
        default="reports/external_radar_hubert_replication.md",
    )
    args = parser.parse_args()

    radar = np.load(args.radar_features)
    teacher = load_teacher_targets(args.teacher_targets)
    radar_ids = radar["sample_ids"].astype(str)
    target_ids = np.asarray(teacher["group_names"]).astype(str)
    target_index = {sample_id: index for index, sample_id in enumerate(target_ids)}
    if len(target_index) != len(target_ids):
        raise ValueError("Teacher target file contains duplicate sample identifiers")
    missing = sorted(set(radar_ids) - set(target_index))
    if missing:
        raise ValueError(f"Teacher targets are missing {len(missing)} radar samples")

    order = np.asarray([target_index[sample_id] for sample_id in radar_ids])
    features = radar["features"].astype(np.float32)
    labels = radar["class_ids"].astype(np.int64)
    teacher_labels = np.asarray(teacher["labels"])[order].astype(np.int64)
    if not np.array_equal(labels, teacher_labels):
        raise ValueError("Radar and teacher class labels are misaligned")
    segments, target_dim = teacher["target_shape"]
    targets = np.asarray(teacher["targets"])[order].reshape(-1, segments, target_dim)
    sessions = radar["session_ids"].astype(str)
    unique_sessions = sorted(np.unique(sessions))
    if len(unique_sessions) != 3:
        raise ValueError(f"Expected three sessions, found {unique_sessions}")

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.perf_counter()
    for fold, test_session in enumerate(unique_sessions):
        val_session = unique_sessions[(fold + 1) % len(unique_sessions)]
        train_session = next(
            session for session in unique_sessions if session not in {test_session, val_session}
        )
        train_mask = sessions == train_session
        val_mask = sessions == val_session
        test_mask = sessions == test_session

        input_mean = features[train_mask].mean(axis=(0, 1), keepdims=True)
        input_std = features[train_mask].std(axis=(0, 1), keepdims=True) + 1e-6
        teacher_center = targets[train_mask].mean(axis=0, keepdims=True)
        centered_targets = targets - teacher_center
        class_prototypes = {
            label: centered_targets[train_mask & (labels == label)].mean(axis=0)
            for label in np.unique(labels[train_mask])
        }
        residual_targets = np.stack(
            [
                target - class_prototypes[label]
                for target, label in zip(centered_targets, labels, strict=True)
            ]
        )

        def prepared(
            mask: np.ndarray, target_values: np.ndarray
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            return (
                (features[mask] - input_mean) / input_std,
                target_values[mask],
                labels[mask],
            )

        train = prepared(train_mask, centered_targets)
        val = prepared(val_mask, centered_targets)
        test = prepared(test_mask, centered_targets)
        seed = args.seed + fold
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = TemporalSensorStudent(
            input_dim=features.shape[2],
            target_dim=target_dim,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
            num_classes=int(labels.max()) + 1,
            num_segments=segments,
        )
        model, best_epoch = _train(
            model,
            train,
            val,
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=seed,
        )
        metrics = _metrics(model, *test, device)
        metrics["class_prototype_cosine"] = _class_prototype_cosine(
            train[1], train[2], test[1], test[2]
        )
        np.random.seed(seed + 1000)
        torch.manual_seed(seed + 1000)
        residual_model = TemporalSensorStudent(
            input_dim=features.shape[2],
            target_dim=target_dim,
            hidden_dim=args.hidden_dim,
            bottleneck_dim=args.bottleneck_dim,
            num_classes=int(labels.max()) + 1,
            num_segments=segments,
        )
        residual_model, residual_best_epoch = _train(
            residual_model,
            prepared(train_mask, residual_targets),
            prepared(val_mask, residual_targets),
            device=device,
            max_epochs=args.max_epochs,
            batch_size=args.batch_size,
            seed=seed + 1000,
        )
        residual_metrics = _metrics(
            residual_model, *prepared(test_mask, residual_targets), device
        )
        metrics.update(
            {
                "residual_best_epoch": residual_best_epoch,
                "residual_segment_cosine": residual_metrics["segment_cosine"],
                "residual_reversed_segment_cosine": residual_metrics[
                    "reversed_segment_cosine"
                ],
                "residual_order_margin_reversed": residual_metrics[
                    "order_margin_reversed"
                ],
                "residual_target_mse": residual_metrics["target_mse"],
            }
        )
        rows.append(
            {
                "fold": fold,
                "train_session": train_session,
                "val_session": val_session,
                "test_session": test_session,
                "best_epoch": best_epoch,
                "num_train": int(train_mask.sum()),
                "num_val": int(val_mask.sum()),
                "num_test": int(test_mask.sum()),
                **metrics,
            }
        )
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_mean": input_mean,
                "input_std": input_std,
                "teacher_center": teacher_center,
                "train_session": train_session,
                "val_session": val_session,
                "test_session": test_session,
            },
            output_dir / f"session_fold_{fold}.pt",
        )
        torch.save(
            {
                "state_dict": residual_model.state_dict(),
                "input_mean": input_mean,
                "input_std": input_std,
                "teacher_center": teacher_center,
                "class_prototypes": class_prototypes,
                "train_session": train_session,
                "val_session": val_session,
                "test_session": test_session,
            },
            output_dir / f"session_fold_{fold}_class_residual.pt",
        )
        elapsed = time.perf_counter() - started
        remaining = elapsed / (fold + 1) * (len(unique_sessions) - fold - 1)
        print(
            f"EXTERNAL_STUDENT fold={fold} test={test_session} "
            f"accuracy={metrics['accuracy']:.3f} cosine={metrics['segment_cosine']:.3f} "
            f"estimated_remaining_seconds={remaining:.1f}",
            flush=True,
        )

    results = pd.DataFrame(rows)
    results_path = Path(args.results_output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    _write_report(Path(args.report_output), results)
    print(f"Saved external replication report to {args.report_output}")


if __name__ == "__main__":
    main()
