#!/usr/bin/env python3
"""Recover class accuracy while retaining ordered temporal HuBERT alignment."""

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

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.models.students.temporal_sensor_student import MultitaskTemporalSensorStudent
from silent_speech_interpretability.models.teachers.teacher_targets import load_teacher_targets


def _index(data, prefix: str) -> dict[tuple[str, str], int]:
    return {
        (str(user), str(group)): index
        for index, (user, group) in enumerate(
            zip(data[f"{prefix}_user_ids"], data[f"{prefix}_group_names"], strict=True)
        )
    }


def _arrays(
    data,
    teacher: dict[str, object],
    modalities: list[str],
    speakers: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[tuple[str, str]]]:
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


def _metrics(
    model: MultitaskTemporalSensorStudent,
    x: np.ndarray,
    targets: np.ndarray,
    labels: np.ndarray,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    with torch.no_grad():
        output = model(torch.tensor(x, dtype=torch.float32, device=device))
        target = F.normalize(torch.tensor(targets, dtype=torch.float32, device=device), p=2, dim=-1)
        true_cosine = F.cosine_similarity(output["target"], target, dim=-1)
        reversed_cosine = F.cosine_similarity(output["target"], target.flip(1), dim=-1)
        shifted_cosine = F.cosine_similarity(output["target"], target.roll(1, dims=1), dim=-1)
    return {
        "accuracy": float((output["logits"].argmax(dim=1).cpu().numpy() == labels).mean()),
        "segment_cosine": float(true_cosine.mean().item()),
        "reversed_segment_cosine": float(reversed_cosine.mean().item()),
        "shifted_segment_cosine": float(shifted_cosine.mean().item()),
        "order_margin_reversed": float((true_cosine - reversed_cosine).mean().item()),
        "target_mse": float(((output["target"] - target) ** 2).sum(dim=-1).mean().item()),
    }


def _make_model(config: dict, input_dim: int, teacher: dict[str, object]) -> MultitaskTemporalSensorStudent:
    return MultitaskTemporalSensorStudent(
        input_dim=input_dim,
        target_dim=int(teacher["target_shape"][1]),
        hidden_dim=int(config["student"]["hidden_dim"]),
        bottleneck_dim=int(config["student"]["bottleneck_dim"]),
        num_classes=int(config["classes"]["num_classes"]),
        num_segments=int(teacher["target_shape"][0]),
    )


def _train_candidate(
    config: dict,
    teacher: dict[str, object],
    train: tuple[np.ndarray, np.ndarray, np.ndarray],
    val: tuple[np.ndarray, np.ndarray, np.ndarray],
    device: torch.device,
    classification_weight: float,
    max_epochs: int,
    batch_size: int,
    seed: int,
) -> tuple[dict[str, torch.Tensor], int, dict[str, float]]:
    torch.manual_seed(seed)
    model = _make_model(config, train[0].shape[2], teacher).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(train[0], dtype=torch.float32),
        torch.tensor(train[1], dtype=torch.float32),
        torch.tensor(train[2], dtype=torch.long),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_epoch = 0
    best_score = -np.inf
    best_metrics: dict[str, float] = {}
    patience = 0
    for epoch in range(1, max_epochs + 1):
        model.train()
        for x, targets, labels in loader:
            x, targets, labels = x.to(device), targets.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(x)
            normalized_targets = F.normalize(targets, p=2, dim=-1)
            alignment = ((output["target"] - normalized_targets) ** 2).sum(dim=-1).mean()
            classification = F.cross_entropy(output["logits"], labels, label_smoothing=0.05)
            (alignment + classification_weight * classification).backward()
            optimizer.step()
        val_metrics = _metrics(model, *val, device)
        joint_score = val_metrics["accuracy"] + val_metrics["segment_cosine"]
        if joint_score > best_score + 1e-4:
            best_score = joint_score
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            best_metrics = val_metrics
            patience = 0
        else:
            patience += 1
            if patience >= 15:
                break
    return best_state, best_epoch, best_metrics


def _tradeoff_figure(
    path: Path,
    results: pd.DataFrame,
    baseline: pd.DataFrame,
    fixed: pd.DataFrame,
) -> None:
    methods = [
        ("Fixed embeddings", float(fixed.accuracy.mean()), float(fixed.segment_cosine.mean())),
        ("Temporal states", float(baseline.accuracy.mean()), float(baseline.segment_cosine.mean())),
        ("Multitask states", float(results.accuracy.mean()), float(results.segment_cosine.mean())),
    ]
    width, height = 900, 380
    left, right, top, bottom = 90, 35, 70, 65
    chart_width, chart_height = width - left - right, height - top - bottom
    items = [
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="24" y="34" font-family="Arial" font-size="20" font-weight="700" fill="#111827">Classification and ordered HuBERT alignment</text>',
    ]
    for tick in range(5):
        value = tick * 0.2
        y = top + chart_height * (1.0 - value)
        items.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        items.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="Arial" font-size="11" fill="#374151">{value:.1f}</text>')
    group_width = chart_width / len(methods)
    for index, (label, accuracy, cosine) in enumerate(methods):
        center = left + group_width * (index + 0.5)
        for offset, value, color in ((-34, accuracy, "#0f766e"), (4, cosine, "#64748b")):
            bar_height = chart_height * value
            items.append(f'<rect x="{center+offset:.1f}" y="{top+chart_height-bar_height:.1f}" width="30" height="{bar_height:.1f}" fill="{color}" rx="2"/>')
            items.append(f'<text x="{center+offset+15:.1f}" y="{top+chart_height-bar_height-6:.1f}" text-anchor="middle" font-family="Arial" font-size="10" fill="#111827">{value:.3f}</text>')
        items.append(f'<text x="{center:.1f}" y="{height-30}" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">{label}</text>')
    items.extend(
        [
            '<rect x="655" y="18" width="12" height="12" fill="#0f766e"/><text x="673" y="29" font-family="Arial" font-size="11" fill="#111827">Accuracy</text>',
            '<rect x="755" y="18" width="12" height="12" fill="#64748b"/><text x="773" y="29" font-family="Arial" font-size="11" fill="#111827">Cosine</text>',
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">\n'
        + "\n".join(items)
        + "\n</svg>\n",
        encoding="utf-8",
    )


def _write_report(
    path: Path,
    figure_path: Path,
    results: pd.DataFrame,
    sweep: pd.DataFrame,
    baseline: pd.DataFrame,
    fixed: pd.DataFrame,
) -> None:
    baseline = baseline.set_index("fold")
    rows = "\n".join(
        f"| {int(row.fold)} | {row.classification_weight:.1f} | {100*row.accuracy:.1f}% | "
        f"{100*baseline.loc[int(row.fold), 'accuracy']:.1f}% | {row.segment_cosine:.3f} | "
        f"{baseline.loc[int(row.fold), 'segment_cosine']:.3f} | {row.reversed_segment_cosine:.3f} | "
        f"{row.order_margin_reversed:+.3f} |"
        for row in results.itertuples(index=False)
    )
    sweep_rows = "\n".join(
        f"| {weight:.1f} | {100*group.val_accuracy.mean():.1f}% | {group.val_segment_cosine.mean():.3f} | "
        f"{group.selected.sum()}/{group.fold.nunique()} |"
        for weight, group in sweep.groupby("classification_weight")
    )
    accuracy_delta = results.accuracy.mean() - baseline.accuracy.mean()
    cosine_delta = results.segment_cosine.mean() - baseline.segment_cosine.mean()
    report = f"""# Multitask Temporal Sensor Student

This experiment gives the temporal sensor student an order-aware utterance classifier
while retaining the four-segment HuBERT alignment branch. Classification-loss weight is
selected independently in each fold using validation speakers only.

![Multitask tradeoff]({figure_path.relative_to(path.parent)})

## Protocol

- Input: fold-specific lip, laser, mmWave, and UWB temporal encoder states.
- Teacher: four ordered, silence-trimmed HuBERT segments.
- Candidate classification weights: {", ".join(f"{value:.1f}" for value in sorted(sweep.classification_weight.unique()))}.
- Selection score: validation accuracy plus validation true-order segment cosine.
- Test speakers remain untouched until after candidate selection.

## Test Results

| Fold | Selected CE Weight | Multitask Accuracy | Previous Accuracy | Multitask Cosine | Previous Cosine | Reversed Cosine | Order Margin |
|---:|---:|---:|---:|---:|---:|---:|---:|
{rows}

## Aggregate

- Multitask accuracy: **{100*results.accuracy.mean():.1f}% +/- {100*results.accuracy.std(ddof=1):.1f}%**.
- Previous temporal-sensor accuracy: **{100*baseline.accuracy.mean():.1f}%**.
- Fixed-embedding temporal-student accuracy: **{100*fixed.accuracy.mean():.1f}%**.
- Accuracy change: **{100*accuracy_delta:+.1f} percentage points**.
- Multitask true-order cosine: **{results.segment_cosine.mean():.3f}**.
- Previous temporal-sensor cosine: **{baseline.segment_cosine.mean():.3f}**.
- Alignment change: **{cosine_delta:+.3f} cosine**.
- Multitask reversed-order cosine: **{results.reversed_segment_cosine.mean():.3f}**.
- Multitask true-versus-reversed margin: **{results.order_margin_reversed.mean():+.3f}**.

## Validation Sweep

| CE Weight | Mean Validation Accuracy | Mean Validation Cosine | Selected Folds |
|---:|---:|---:|---:|
{sweep_rows}

The loss-weight sweep and early stopping use validation speakers, not test results. As
before, four relative-time regions show ordered evidence but do not establish frame-exact
or phoneme-level synchronization.
"""
    path.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--activations-dir", default="artifacts/activations/temporal_sensors")
    parser.add_argument("--teacher-targets", default="artifacts/teacher_targets/facebook_hubert-base-ls960_temporal4_targets.npz")
    parser.add_argument("--modalities", default="lip,laser,mmwave,uwb")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--classification-weights", default="0.2,0.5,1.0,2.0")
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--output-dir", default="artifacts/students/temporal_sensor_multitask_cv")
    parser.add_argument("--output", default="reports/results/temporal_sensor_multitask_cv.csv")
    parser.add_argument("--sweep-output", default="reports/results/temporal_sensor_multitask_sweep.csv")
    parser.add_argument("--baseline-results", default="reports/results/temporal_sensor_student_cv.csv")
    parser.add_argument("--fixed-results", default="reports/results/hubert_temporal_teacher_student_cv.csv")
    parser.add_argument("--report-output", default="reports/temporal_sensor_multitask.md")
    parser.add_argument("--figure-output", default="reports/figures/temporal_sensor_multitask_tradeoff.svg")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    if args.summarize_only:
        results = pd.read_csv(args.output)
        sweep = pd.read_csv(args.sweep_output)
        baseline = pd.read_csv(args.baseline_results)
        fixed = pd.read_csv(args.fixed_results)
        figure_path = Path(args.figure_output)
        _tradeoff_figure(figure_path, results, baseline, fixed)
        _write_report(Path(args.report_output), figure_path, results, sweep, baseline, fixed)
        print(f"Saved multitask temporal report to {args.report_output}")
        return

    config = load_config(args.config)
    teacher = load_teacher_targets(args.teacher_targets)
    modalities = [value.strip() for value in args.modalities.split(",") if value.strip()]
    folds_requested = {int(value) for value in args.folds.split(",") if value.strip()}
    weights = [float(value) for value in args.classification_weights.split(",") if value.strip()]
    seed_paths, _ = resolve_embedding_paths(
        config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"]
    )
    manifest = build_manifest(seed_paths)
    folds = [
        fold
        for fold in make_speaker_kfold_splits(
            manifest, config["splits"]["num_folds"], config["project"]["seed"]
        )
        if int(fold["fold"]) in folds_requested
    ]
    if args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS unavailable; falling back to CPU", flush=True)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_rows: list[dict[str, float | int]] = []
    sweep_rows: list[dict[str, float | int | bool]] = []
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
        prepared = [
            ((x - input_mean) / input_std, targets - teacher_center, labels)
            for x, targets, labels, _pairs in (train, val, test)
        ]
        seed = int(config["project"]["seed"]) + fold_id
        candidates = []
        for weight in weights:
            state, best_epoch, val_metrics = _train_candidate(
                config,
                teacher,
                prepared[0],
                prepared[1],
                device,
                weight,
                args.max_epochs,
                args.batch_size,
                seed,
            )
            joint_score = val_metrics["accuracy"] + val_metrics["segment_cosine"]
            candidates.append((joint_score, weight, state, best_epoch, val_metrics))
        selected = max(candidates, key=lambda item: item[0])
        _score, selected_weight, selected_state, best_epoch, selected_val = selected
        for score, weight, _state, candidate_epoch, val_metrics in candidates:
            sweep_rows.append(
                {
                    "fold": fold_id,
                    "classification_weight": weight,
                    "best_epoch": candidate_epoch,
                    "val_joint_score": score,
                    "val_accuracy": val_metrics["accuracy"],
                    "val_segment_cosine": val_metrics["segment_cosine"],
                    "val_order_margin_reversed": val_metrics["order_margin_reversed"],
                    "selected": weight == selected_weight,
                }
            )
        model = _make_model(config, prepared[0][0].shape[2], teacher).to(device)
        model.load_state_dict(selected_state)
        test_metrics = _metrics(model, *prepared[2], device)
        result_rows.append(
            {
                "fold": fold_id,
                "classification_weight": selected_weight,
                "best_epoch": best_epoch,
                "num_train": len(train[3]),
                "num_val": len(val[3]),
                "num_test": len(test[3]),
                "val_accuracy": selected_val["accuracy"],
                "val_segment_cosine": selected_val["segment_cosine"],
                **test_metrics,
            }
        )
        torch.save(
            {
                "model_type": "multitask_temporal_sensor",
                "state_dict": selected_state,
                "modalities": modalities,
                "classification_weight": selected_weight,
                "input_mean": input_mean,
                "input_std": input_std,
                "teacher_center": teacher_center,
                "input_dim": prepared[0][0].shape[2],
                "target_dim": int(teacher["target_shape"][1]),
                "hidden_dim": int(config["student"]["hidden_dim"]),
                "bottleneck_dim": int(config["student"]["bottleneck_dim"]),
                "num_classes": int(config["classes"]["num_classes"]),
                "num_segments": int(teacher["target_shape"][0]),
            },
            output_dir / f"fold_{fold_id}_temporal_sensor_multitask.pt",
        )
        elapsed = time.perf_counter() - started
        remaining = elapsed / position * (len(folds) - position)
        print(
            f"MULTITASK_TEMPORAL_CV fold={fold_id} weight={selected_weight:.1f} "
            f"accuracy={test_metrics['accuracy']:.3f} cosine={test_metrics['segment_cosine']:.3f} "
            f"estimated_remaining_seconds={remaining:.1f}",
            flush=True,
        )

    results = pd.DataFrame(result_rows).sort_values("fold")
    sweep = pd.DataFrame(sweep_rows).sort_values(["fold", "classification_weight"])
    output = Path(args.output)
    sweep_output = Path(args.sweep_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output, index=False)
    sweep.to_csv(sweep_output, index=False)
    baseline = pd.read_csv(args.baseline_results)
    fixed = pd.read_csv(args.fixed_results)
    figure_path = Path(args.figure_output)
    _tradeoff_figure(figure_path, results, baseline, fixed)
    _write_report(Path(args.report_output), figure_path, results, sweep, baseline, fixed)
    print(f"Saved multitask temporal report to {args.report_output}")


if __name__ == "__main__":
    main()
