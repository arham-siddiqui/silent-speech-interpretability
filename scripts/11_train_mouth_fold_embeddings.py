#!/usr/bin/env python3
"""Train a fold-specific mouth projection encoder and extract embeddings."""

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
from silent_speech_interpretability.models.encoders.mouth_video import (
    MouthProjectionHead,
    MouthTrainingConfig,
    extract_mouth_embeddings,
    load_mouth_csv,
    make_mouth_dataloaders,
    resolve_mouth_csv,
    save_label_map,
    train_mouth_model,
)


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _training_config(config: dict, args: argparse.Namespace) -> MouthTrainingConfig:
    raw = config.get("mouth_encoder", {})
    return MouthTrainingConfig(
        input_dim=int(raw.get("input_dim", 512)),
        feature_dim=int(raw.get("feature_dim", 256)),
        embedding_dim=int(raw.get("embedding_dim", 128)),
        dropout=float(raw.get("dropout", 0.3)),
        batch_size=int(args.batch_size or raw.get("batch_size", 64)),
        lr=float(raw.get("lr", 3e-4)),
        max_epochs=int(args.max_epochs or raw.get("max_epochs", 60)),
        patience=int(raw.get("patience", 20)),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--mouth-csv", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    seed_paths, _sources = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    fold = next((item for item in folds if int(item["fold"]) == args.fold), None)
    if fold is None:
        raise ValueError(f"Fold {args.fold} not found.")

    csv_path = resolve_mouth_csv(args.mouth_csv or config["data"].get("mouth_csv"), config["data"].get("rvtall_base"))
    payload = load_mouth_csv(csv_path)
    users = [str(user) for user in payload["users"]]
    train_set = {str(speaker) for speaker in fold["train_speakers"]}
    val_set = {str(speaker) for speaker in fold["val_speakers"]}
    test_set = {str(speaker) for speaker in fold["test_speakers"]}
    print(
        f"Fold {args.fold}: {sum(user in train_set for user in users)} train | "
        f"{sum(user in val_set for user in users)} val | {sum(user in test_set for user in users)} test mouth candidates",
        flush=True,
    )
    print(f"Classes: {len(payload['label_map'])}", flush=True)
    print(f"CSV: {csv_path}", flush=True)
    if args.dry_run:
        return

    train_config = _training_config(config, args)
    device = torch.device(args.device) if args.device else _default_device()
    print(f"Device: {device}", flush=True)
    train_loader, val_loader = make_mouth_dataloaders(payload, fold["train_speakers"], fold["val_speakers"], train_config)

    output_paths = expected_fold_embedding_paths(
        config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv"),
        args.fold,
        modalities=("mouth",),
    )
    embedding_path = output_paths["mouth"]
    fold_dir = embedding_path.parent
    checkpoint_path = fold_dir / "mouth_projection_model.pt"
    label_map_path = fold_dir / "mouth_label_map.json"
    save_label_map(payload["label_map"], label_map_path)

    model = MouthProjectionHead(
        num_classes=len(payload["label_map"]),
        input_dim=train_config.input_dim,
        feature_dim=train_config.feature_dim,
        embedding_dim=train_config.embedding_dim,
        dropout=train_config.dropout,
    ).to(device)
    model = train_mouth_model(model, train_loader, val_loader, train_config, checkpoint_path, device)
    count = extract_mouth_embeddings(model, payload, embedding_path, device, batch_size=max(train_config.batch_size, 128))
    metadata_path, metadata = load_or_create_fold_metadata(config, fold)
    mark_modality_complete(
        metadata_path,
        metadata,
        "mouth",
        embedding_path,
        checkpoint_path,
        label_map_path,
        {
            "max_epochs": train_config.max_epochs,
            "batch_size": train_config.batch_size,
            "lr": train_config.lr,
            "feature_dim": train_config.feature_dim,
            "embedding_dim": train_config.embedding_dim,
            "dropout": train_config.dropout,
        },
    )
    print(f"Saved {count} mouth embeddings to {embedding_path}")
    print(f"Updated metadata at {metadata_path}")


if __name__ == "__main__":
    main()
