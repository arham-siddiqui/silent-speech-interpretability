#!/usr/bin/env python3
"""Train a silent-sensor student against fixed teacher targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.embeddings import load_embedding_repetitions, mean_eval_arrays
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.evals.true_cv import configured_fold_embedding_paths
from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent
from silent_speech_interpretability.models.teachers.teacher_targets import common_teacher_pairs, load_teacher_targets, teacher_arrays
from silent_speech_interpretability.training.train_articulatory_student import (
    StudentTrainConfig,
    evaluate_student,
    make_student_arrays,
    train_student,
)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _arrays_for_pairs(payloads: dict[str, dict[str, object]], teacher: dict[str, object], modalities: list[str], pairs: list[tuple[str, str]]):
    modality_arrays = {}
    for modality in modalities:
        x, _labels = mean_eval_arrays(payloads[modality], pairs)
        modality_arrays[modality] = x
    targets, labels = teacher_arrays(teacher, pairs)
    return make_student_arrays(modality_arrays, targets, labels, modalities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--teacher-targets", default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--modalities", default=None, help="Comma-separated modalities; defaults to fusion modalities.")
    parser.add_argument("--max-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default=None)
    parser.add_argument("--output-dir", default="artifacts/students")
    args = parser.parse_args()

    config = load_config(args.config)
    teacher_path = Path(args.teacher_targets or Path(config["data"]["teacher_targets_dir"]) / "synthetic_audio_teacher_targets.npz")
    teacher = load_teacher_targets(teacher_path)

    seed_paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    fold = next(item for item in folds if int(item["fold"]) == args.fold)

    paths = configured_fold_embedding_paths(config.get("true_encoder_cv", {}), args.fold)
    payloads = {modality: load_embedding_repetitions(path) for modality, path in paths.items()}
    if args.modalities:
        modalities = [item.strip() for item in args.modalities.split(",") if item.strip()]
    else:
        excluded = set(config.get("fusion", {}).get("excluded_modalities", []))
        modalities = [modality for modality in ["lip", "uwb", "mmwave", "laser", "mouth"] if modality in payloads and modality not in excluded]

    common_payloads = {modality: payloads[modality] for modality in modalities}
    train_pairs = common_teacher_pairs(common_payloads, teacher, fold["train_speakers"])
    val_pairs = common_teacher_pairs(common_payloads, teacher, fold["val_speakers"])
    test_pairs = common_teacher_pairs(common_payloads, teacher, fold["test_speakers"])
    if not train_pairs or not val_pairs or not test_pairs:
        raise RuntimeError(f"Empty student split: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    train_arrays = _arrays_for_pairs(common_payloads, teacher, modalities, train_pairs)
    val_arrays = _arrays_for_pairs(common_payloads, teacher, modalities, val_pairs)
    test_arrays = _arrays_for_pairs(common_payloads, teacher, modalities, test_pairs)

    device = torch.device(args.device) if args.device else _default_device()
    student_config = config.get("student", {})
    train_config = StudentTrainConfig(
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        lr=float(student_config.get("lr", 3e-4)),
        ce_weight=float(student_config.get("loss_weights", {}).get("ce", 0.2)),
    )
    model = ArticulatoryStudent(
        modalities=modalities,
        embedding_dim=int(config["modalities"][modalities[0]].get("embedding_dim", 128)),
        hidden_dim=int(student_config.get("hidden_dim", 256)),
        bottleneck_dim=int(student_config.get("bottleneck_dim", 64)),
        target_dim=int(teacher["target_dim"]),
        num_classes=int(config["classes"]["num_classes"]),
    )
    model, val_metrics = train_student(model, train_arrays, val_arrays, train_config, device)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(
            torch.tensor(test_arrays[0], dtype=torch.float32),
            torch.tensor(test_arrays[1], dtype=torch.float32),
            torch.tensor(test_arrays[2], dtype=torch.long),
        ),
        batch_size=args.batch_size,
        shuffle=False,
    )
    test_metrics = evaluate_student(model, test_loader, device, train_config.ce_weight)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / f"fold_{args.fold}_teacher_student.pt"
    metrics_path = output_dir / f"fold_{args.fold}_teacher_student_metrics.json"
    torch.save({"state_dict": model.state_dict(), "modalities": modalities, "teacher_targets": str(teacher_path)}, checkpoint_path)
    metrics = {
        "fold": args.fold,
        "modalities": modalities,
        "teacher_targets": str(teacher_path),
        "num_train": len(train_pairs),
        "num_val": len(val_pairs),
        "num_test": len(test_pairs),
        "val": val_metrics,
        "test": test_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved checkpoint to {checkpoint_path}")
    print(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
