#!/usr/bin/env python3
"""Train a fold-specific UWB encoder and extract embeddings."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.evals.fold_metadata import load_or_create_fold_metadata, mark_modality_complete
from silent_speech_interpretability.evals.true_cv import expected_fold_embedding_paths
from silent_speech_interpretability.models.encoders.uwb import (
    UWBEncoderV2,
    UWBTrainingConfig,
    build_uwb_sample_list,
    extract_uwb_embeddings,
    make_uwb_dataloaders,
    save_label_map,
    train_uwb_model,
)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _training_config(config: dict, args: argparse.Namespace) -> UWBTrainingConfig:
    raw = config.get("uwb_encoder", {})
    return UWBTrainingConfig(
        batch_size=int(args.batch_size or raw.get("batch_size", 16)),
        lr=float(raw.get("lr", 3e-4)),
        max_epochs=int(args.max_epochs or raw.get("max_epochs", 80)),
        patience=int(raw.get("patience", 25)),
        hidden_size=int(raw.get("hidden_size", 128)),
        embedding_dim=int(raw.get("embedding_dim", 128)),
        dropout=float(raw.get("dropout", 0.3)),
        lambda_dann=float(raw.get("lambda_dann", 0.3)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--rvtall-base", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
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

    samples = build_uwb_sample_list(rvtall_base)
    label_map = {label: idx for idx, label in enumerate(sorted({sample["label_str"] for sample in samples}))}
    train_set = {str(speaker) for speaker in fold["train_speakers"]}
    val_set = {str(speaker) for speaker in fold["val_speakers"]}
    test_set = {str(speaker) for speaker in fold["test_speakers"]}
    print(f"Fold {args.fold}: {sum(s['user_id'] in train_set for s in samples)} train | {sum(s['user_id'] in val_set for s in samples)} val | {sum(s['user_id'] in test_set for s in samples)} test UWB candidates", flush=True)
    print(f"Classes: {len(label_map)}", flush=True)
    if args.dry_run:
        return

    train_config = _training_config(config, args)
    device = torch.device(args.device) if args.device else _default_device()
    print(f"Device: {device}", flush=True)
    train_loader, val_loader, speaker_map = make_uwb_dataloaders(samples, fold["train_speakers"], fold["val_speakers"], label_map, train_config)

    output_paths = expected_fold_embedding_paths(config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv"), args.fold, modalities=("uwb",))
    embedding_path = output_paths["uwb"]
    fold_dir = embedding_path.parent
    checkpoint_path = fold_dir / "uwb_cnn_lstm_model.pt"
    label_map_path = fold_dir / "uwb_label_map.json"
    save_label_map(label_map, label_map_path)

    model = UWBEncoderV2(
        num_classes=len(label_map),
        num_speakers=len(speaker_map),
        hidden_size=train_config.hidden_size,
        embedding_dim=train_config.embedding_dim,
        dropout=train_config.dropout,
    ).to(device)
    model = train_uwb_model(model, train_loader, val_loader, train_config, checkpoint_path, device)
    count = extract_uwb_embeddings(model, samples, label_map, embedding_path, device)
    metadata_path, metadata = load_or_create_fold_metadata(config, fold)
    mark_modality_complete(
        metadata_path,
        metadata,
        "uwb",
        embedding_path,
        checkpoint_path,
        label_map_path,
        {
            "max_epochs": train_config.max_epochs,
            "batch_size": train_config.batch_size,
            "lr": train_config.lr,
            "hidden_size": train_config.hidden_size,
            "embedding_dim": train_config.embedding_dim,
            "dropout": train_config.dropout,
            "lambda_dann": train_config.lambda_dann,
        },
    )
    print(f"Saved {count} UWB embeddings to {embedding_path}")
    print(f"Updated metadata at {metadata_path}")


if __name__ == "__main__":
    main()
