"""Mouth-frame projection encoder for fold-specific embedding extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class MouthTrainingConfig:
    input_dim: int = 512
    feature_dim: int = 256
    embedding_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 64
    lr: float = 3e-4
    max_epochs: int = 60
    patience: int = 20
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


def resolve_mouth_csv(path: str | Path | None, rvtall_base: str | Path | None = None) -> Path:
    candidates = []
    if path:
        candidates.append(Path(path).expanduser())
    if rvtall_base:
        base = Path(rvtall_base).expanduser()
        candidates.extend(
            [
                base / "mouth_frame_embeddings.csv",
                base.parent.parent.parent / "mouth_frame_embeddings.csv",
                base.parents[3] / "mouth_frame_embeddings.csv" if len(base.parents) > 3 else base / "mouth_frame_embeddings.csv",
            ]
        )
    candidates.extend([Path("mouth_frame_embeddings.csv"), Path("../silentSpeech/mouth_frame_embeddings.csv")])
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except FileNotFoundError:
            resolved = candidate
        if resolved.exists():
            return resolved
    raise FileNotFoundError("Could not find mouth_frame_embeddings.csv; pass --mouth-csv or set data.mouth_csv.")


def load_mouth_csv(path: str | Path) -> dict[str, object]:
    df = pd.read_csv(path)
    embed_cols = [f"embed_{i}" for i in range(512)]
    missing = [col for col in embed_cols + ["label_name", "participant", "video_name"] if col not in df.columns]
    if missing:
        raise ValueError(f"Mouth CSV is missing columns: {missing[:8]}")
    labels_raw = df["label_name"].astype(str).to_numpy()
    label_map = {label: idx for idx, label in enumerate(sorted(set(labels_raw)))}
    return {
        "features": df[embed_cols].to_numpy(dtype=np.float32),
        "labels": np.asarray([label_map[label] for label in labels_raw], dtype=np.int64),
        "users": df["participant"].astype(str).to_numpy(),
        "label_names": labels_raw,
        "video_names": df["video_name"].astype(str).to_numpy(),
        "label_map": label_map,
    }


class MouthProjectionHead(nn.Module):
    def __init__(self, num_classes: int, input_dim: int = 512, feature_dim: int = 256, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feature_dim),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        embedding = self.proj(x)
        logits = self.classifier(embedding)
        return logits, F.normalize(embedding, p=2, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _logits, embedding = self.forward(x)
        return embedding


def make_mouth_dataloaders(payload: dict[str, object], train_speakers: list[int], val_speakers: list[int], config: MouthTrainingConfig) -> tuple[DataLoader, DataLoader]:
    features = torch.tensor(payload["features"], dtype=torch.float32)
    labels = torch.tensor(payload["labels"], dtype=torch.long)
    users = np.asarray(payload["users"]).astype(str)
    train_users = {str(speaker) for speaker in train_speakers}
    val_users = {str(speaker) for speaker in val_speakers}
    train_idx = np.asarray([idx for idx, user in enumerate(users) if user in train_users], dtype=np.int64)
    val_idx = np.asarray([idx for idx, user in enumerate(users) if user in val_users], dtype=np.int64)
    if len(train_idx) == 0 or len(val_idx) == 0:
        raise RuntimeError(f"Empty mouth split: train={len(train_idx)}, val={len(val_idx)}")
    train_loader = DataLoader(TensorDataset(features[train_idx], labels[train_idx]), batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(features[val_idx], labels[val_idx]), batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_mouth_model(model: MouthProjectionHead, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for features, labels in loader:
        features, labels = features.to(device), labels.to(device)
        logits, _embedding = model(features)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_mouth_model(model: MouthProjectionHead, train_loader: DataLoader, val_loader: DataLoader, config: MouthTrainingConfig, checkpoint_path: str | Path, device: torch.device) -> MouthProjectionHead:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0
    patience_count = 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for features, labels in train_loader:
            features, labels = features.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _embedding = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
        val_loss, val_acc = evaluate_mouth_model(model, val_loader, criterion, device)
        scheduler.step()
        print(
            f"Epoch {epoch:03d}/{config.max_epochs} train_loss={total_loss / max(total, 1):.4f} "
            f"train_acc={correct / max(total, 1):.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience_count += 1
            if patience_count >= config.patience:
                break
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


@torch.no_grad()
def extract_mouth_embeddings(model: MouthProjectionHead, payload: dict[str, object], output_path: str | Path, device: torch.device, batch_size: int = 128) -> int:
    model.eval()
    features = torch.tensor(payload["features"], dtype=torch.float32)
    embeddings = []
    for start in range(0, len(features), batch_size):
        chunk = features[start : start + batch_size].to(device)
        embeddings.append(model.encode(chunk).cpu().numpy())
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.concatenate(embeddings, axis=0).astype(np.float32),
        labels=np.asarray(payload["labels"], dtype=np.int32),
        users=np.asarray(payload["users"]),
        label_names=np.asarray(payload["label_names"]),
        video_names=np.asarray(payload["video_names"]),
    )
    return int(len(features))


def save_label_map(label_map: dict[str, int], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
