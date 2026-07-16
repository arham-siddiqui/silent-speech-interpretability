#!/usr/bin/env python3
"""Train a fold-specific laser encoder and extract fold embeddings."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.evals.true_cv import expected_fold_embedding_paths, metadata_path_for_fold
from silent_speech_interpretability.models.encoders.laser import (
    LaserCNNLSTMEncoder,
    LaserTrainingConfig,
    build_laser_sample_list,
    extract_laser_embeddings,
    make_laser_dataloaders,
    save_label_map,
    train_laser_model,
)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _training_config(config: dict, args: argparse.Namespace) -> LaserTrainingConfig:
    raw = config.get("laser_encoder", {})
    return LaserTrainingConfig(
        hidden_size=int(raw.get("hidden_size", 128)),
        num_layers=int(raw.get("num_layers", 2)),
        embedding_dim=int(raw.get("embedding_dim", 128)),
        dropout=float(raw.get("dropout", 0.3)),
        batch_size=int(args.batch_size or raw.get("batch_size", 32)),
        lr=float(raw.get("lr", 3e-4)),
        max_epochs=int(args.max_epochs or raw.get("max_epochs", 60)),
        patience=int(raw.get("patience", 20)),
    )


def _metadata(config: dict, fold: dict) -> tuple[Path, dict]:
    embeddings_dir = Path(config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv"))
    path = metadata_path_for_fold(embeddings_dir, int(fold["fold"]))
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {
            "fold": int(fold["fold"]),
            "train_speakers": fold["train_speakers"],
            "val_speakers": fold["val_speakers"],
            "test_speakers": fold["test_speakers"],
            "modalities": ["lip", "mouth", "uwb", "mmwave", "laser"],
            "status": "planned",
        }
    payload.setdefault("completed_modalities", [])
    return path, payload


def _mark_laser_complete(
    metadata_path: Path,
    metadata: dict,
    embedding_path: Path,
    checkpoint_path: Path,
    label_map_path: Path,
    train_config: LaserTrainingConfig,
) -> None:
    completed = set(metadata.get("completed_modalities", []))
    completed.add("laser")
    metadata["completed_modalities"] = sorted(completed)
    metadata["laser_embedding_path"] = str(embedding_path)
    metadata["laser_checkpoint_path"] = str(checkpoint_path)
    metadata["laser_label_map_path"] = str(label_map_path)
    metadata["laser_training"] = {
        "max_epochs": train_config.max_epochs,
        "batch_size": train_config.batch_size,
        "lr": train_config.lr,
        "hidden_size": train_config.hidden_size,
        "num_layers": train_config.num_layers,
        "embedding_dim": train_config.embedding_dim,
        "dropout": train_config.dropout,
        "note": "Short max_epochs values are smoke tests, not final scientific fold embeddings.",
    }
    expected = set(metadata.get("modalities", []))
    metadata["status"] = "completed" if expected and expected.issubset(completed) else "partial"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--rvtall-base", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from a fold training-state checkpoint or warm-start from model weights.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    rvtall_base = args.rvtall_base or config["data"].get("rvtall_base")
    if not rvtall_base:
        raise RuntimeError("Set data.rvtall_base in config or pass --rvtall-base.")

    seed_paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    fold = next((item for item in folds if int(item["fold"]) == args.fold), None)
    if fold is None:
        raise ValueError(f"Fold {args.fold} not found.")

    samples = build_laser_sample_list(rvtall_base)
    label_map = {label: idx for idx, label in enumerate(sorted({sample["label_str"] for sample in samples}))}
    train_set = {str(speaker) for speaker in fold["train_speakers"]}
    val_set = {str(speaker) for speaker in fold["val_speakers"]}
    test_set = {str(speaker) for speaker in fold["test_speakers"]}
    train_count = sum(sample["user_id"] in train_set for sample in samples)
    val_count = sum(sample["user_id"] in val_set for sample in samples)
    test_count = sum(sample["user_id"] in test_set for sample in samples)
    print(f"Fold {args.fold}: {train_count} train | {val_count} val | {test_count} test laser candidates", flush=True)
    print(f"Classes: {len(label_map)}", flush=True)
    if args.dry_run:
        return

    train_config = _training_config(config, args)
    device = torch.device(args.device) if args.device else _default_device()
    print(f"Device: {device}", flush=True)
    train_loader, val_loader = make_laser_dataloaders(samples, fold["train_speakers"], fold["val_speakers"], label_map, train_config)

    output_paths = expected_fold_embedding_paths(
        config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv"),
        args.fold,
        modalities=("laser",),
    )
    embedding_path = output_paths["laser"]
    fold_dir = embedding_path.parent
    checkpoint_path = fold_dir / "laser_cnn_lstm_model.pt"
    label_map_path = fold_dir / "laser_label_map.json"
    save_label_map(label_map, label_map_path)

    model = LaserCNNLSTMEncoder(
        num_classes=len(label_map),
        hidden_size=train_config.hidden_size,
        num_layers=train_config.num_layers,
        embedding_dim=train_config.embedding_dim,
        dropout=train_config.dropout,
    ).to(device)
    model = train_laser_model(model, train_loader, val_loader, train_config, checkpoint_path, device, resume=args.resume)
    count = extract_laser_embeddings(model, samples, label_map, embedding_path, device, batch_size=max(train_config.batch_size, 64))
    metadata_path, metadata = _metadata(config, fold)
    _mark_laser_complete(metadata_path, metadata, embedding_path, checkpoint_path, label_map_path, train_config)
    print(f"Saved {count} laser embeddings to {embedding_path}")
    print(f"Updated metadata at {metadata_path}")


if __name__ == "__main__":
    main()
