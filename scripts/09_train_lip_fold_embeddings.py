#!/usr/bin/env python3
"""Train a fold-specific lip landmark encoder and extract fold embeddings."""

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
from silent_speech_interpretability.models.encoders.lip import (
    LipLSTMV2,
    LipTrainingConfig,
    build_lip_sample_list,
    extract_lip_embeddings,
    make_lip_dataloaders,
    save_label_map,
    train_lip_model,
)


def _training_config(config: dict, args: argparse.Namespace) -> LipTrainingConfig:
    raw = config.get("lip_encoder", {})
    return LipTrainingConfig(
        hidden_size=int(raw.get("hidden_size", 256)),
        num_layers=int(raw.get("num_layers", 2)),
        embedding_dim=int(raw.get("embedding_dim", 128)),
        dropout=float(raw.get("dropout", 0.3)),
        batch_size=int(args.batch_size or raw.get("batch_size", 32)),
        lr=float(raw.get("lr", 3e-4)),
        max_epochs=int(args.max_epochs or raw.get("max_epochs", 80)),
        patience=int(raw.get("patience", 25)),
    )


def _load_or_create_metadata(config: dict, fold: dict) -> tuple[Path, dict]:
    embeddings_dir = Path(config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv"))
    path = metadata_path_for_fold(embeddings_dir, int(fold["fold"]))
    if path.exists():
        metadata = json.loads(path.read_text(encoding="utf-8"))
    else:
        metadata = {
            "fold": int(fold["fold"]),
            "train_speakers": fold["train_speakers"],
            "val_speakers": fold["val_speakers"],
            "test_speakers": fold["test_speakers"],
            "modalities": ["lip", "mouth", "uwb", "mmwave", "laser"],
            "status": "planned",
        }
    metadata.setdefault("completed_modalities", [])
    return path, metadata


def _mark_lip_complete(metadata_path: Path, metadata: dict, embedding_path: Path, checkpoint_path: Path, label_map_path: Path) -> None:
    completed = set(metadata.get("completed_modalities", []))
    completed.add("lip")
    metadata["completed_modalities"] = sorted(completed)
    metadata["lip_embedding_path"] = str(embedding_path)
    metadata["lip_checkpoint_path"] = str(checkpoint_path)
    metadata["lip_label_map_path"] = str(label_map_path)
    expected = set(metadata.get("modalities", []))
    metadata["status"] = "completed" if expected and expected.issubset(completed) else "partial"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--rvtall-base", default=None, help="Path to RVTALL/Processed_cut_data or kinect_processed.")
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

    samples = build_lip_sample_list(rvtall_base)
    label_map = {label: idx for idx, label in enumerate(sorted({sample["label_str"] for sample in samples}))}
    train_count = sum(sample["user_id"] in {str(s) for s in fold["train_speakers"]} for sample in samples)
    val_count = sum(sample["user_id"] in {str(s) for s in fold["val_speakers"]} for sample in samples)
    test_count = sum(sample["user_id"] in {str(s) for s in fold["test_speakers"]} for sample in samples)
    print(f"Fold {args.fold}: {train_count} train | {val_count} val | {test_count} test lip candidates")
    print(f"Classes: {len(label_map)}")
    if args.dry_run:
        return

    train_config = _training_config(config, args)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    train_loader, val_loader, speaker_map = make_lip_dataloaders(
        samples,
        fold["train_speakers"],
        fold["val_speakers"],
        label_map,
        train_config,
    )

    output_paths = expected_fold_embedding_paths(
        config.get("true_encoder_cv", {}).get("embeddings_dir", "artifacts/embeddings/speaker_cv"),
        args.fold,
        modalities=("lip",),
    )
    embedding_path = output_paths["lip"]
    fold_dir = embedding_path.parent
    checkpoint_path = fold_dir / "lip_lstm_model.pt"
    label_map_path = fold_dir / "lip_label_map.json"
    save_label_map(label_map, label_map_path)

    model = LipLSTMV2(
        input_size=80,
        num_classes=len(label_map),
        num_speakers=len(speaker_map),
        hidden_size=train_config.hidden_size,
        num_layers=train_config.num_layers,
        embedding_dim=train_config.embedding_dim,
        dropout=train_config.dropout,
    ).to(device)
    model = train_lip_model(model, train_loader, val_loader, train_config, checkpoint_path, device)
    count = extract_lip_embeddings(model, samples, label_map, embedding_path, device)
    metadata_path, metadata = _load_or_create_metadata(config, fold)
    _mark_lip_complete(metadata_path, metadata, embedding_path, checkpoint_path, label_map_path)
    print(f"Saved {count} lip embeddings to {embedding_path}")
    print(f"Updated metadata at {metadata_path}")


if __name__ == "__main__":
    main()
