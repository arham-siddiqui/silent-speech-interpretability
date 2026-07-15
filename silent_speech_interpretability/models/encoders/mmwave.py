"""mmWave/FMCW radar CNN+LSTM encoder and embedding extraction."""

from __future__ import annotations

import glob
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

N_RANGE_BINS = 513


@dataclass
class MmwaveTrainingConfig:
    batch_size: int = 16
    lr: float = 3e-4
    max_epochs: int = 60
    patience: int = 20
    hidden_size: int = 128
    num_layers: int = 2
    embedding_dim: int = 128
    dropout: float = 0.3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


def natural_sort_key(path: str | Path) -> list[object]:
    name = Path(path).name
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def resolve_mmwave_root(rvtall_base: str | Path) -> Path:
    base = Path(rvtall_base).expanduser().resolve()
    if base.name == "radar_processed":
        return base
    if (base / "radar_processed").is_dir():
        return base / "radar_processed"
    raise FileNotFoundError(f"Could not find radar_processed under {base}")


def load_mmwave_rtm(path: str | Path) -> np.ndarray | None:
    rtm = np.load(path).astype(np.float32)
    if rtm.ndim != 2 or rtm.shape[0] != N_RANGE_BINS or rtm.shape[1] < 10:
        return None
    rtm = np.log1p(rtm)
    rtm = rtm - rtm.mean(axis=1, keepdims=True)
    std = rtm.std()
    if std > 1e-8:
        rtm = rtm / std
    return rtm.T.astype(np.float32)


def build_mmwave_sample_list(rvtall_base: str | Path) -> list[dict[str, str]]:
    root = resolve_mmwave_root(rvtall_base)
    samples = []
    for user_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=natural_sort_key):
        user_id = user_dir.name
        for group_prefix in ("sentences", "vowel", "word"):
            for group_dir in sorted(glob.glob(str(user_dir / f"{group_prefix}*")), key=natural_sort_key):
                group_path = Path(group_dir)
                for npy_path in sorted(group_path.glob("*.npy"), key=natural_sort_key):
                    samples.append(
                        {
                            "user_id": user_id,
                            "group_name": group_path.name,
                            "sample_name": npy_path.name,
                            "path": str(npy_path),
                            "label_str": group_path.name,
                        }
                    )
    return samples


class MmwaveDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]], label_map: dict[str, int], augment: bool = False):
        self.augment = augment
        self.items = []
        for sample in samples:
            try:
                shape = np.load(sample["path"], mmap_mode="r").shape
            except Exception:
                continue
            if len(shape) == 2 and shape[0] == N_RANGE_BINS and shape[1] >= 10:
                self.items.append((sample["path"], label_map[sample["label_str"]], sample))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, label, sample = self.items[idx]
        rtm = load_mmwave_rtm(path)
        if rtm is None:
            raise RuntimeError(f"Invalid mmWave RTM after validation: {path}")
        if self.augment:
            if len(rtm) > 20:
                keep = int(np.random.uniform(0.8, 1.0) * len(rtm))
                start = np.random.randint(0, len(rtm) - keep + 1)
                rtm = rtm[start : start + keep]
            rtm = rtm + np.random.normal(0, 0.02, rtm.shape).astype(np.float32)
        return torch.from_numpy(rtm.astype(np.float32)), label, sample


def collate_mmwave_batch(batch):
    rtms, labels, samples = zip(*batch)
    lengths = torch.tensor([len(rtm) for rtm in rtms], dtype=torch.long)
    padded = pad_sequence(rtms, batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long), list(samples)


class MmwaveCNNLSTMEncoder(nn.Module):
    TIME_STRIDES = (2, 2, 2, 1)

    def __init__(self, num_classes: int, hidden_size: int = 128, num_layers: int = 2, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        channels = (16, 32, 64, 128)
        layers = []
        in_channels = 1
        for out_channels, time_stride in zip(channels, self.TIME_STRIDES):
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(2, time_stride), padding=(1, 1)),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim), nn.LayerNorm(embedding_dim))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _time_out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        output = lengths.float()
        for stride in self.TIME_STRIDES:
            output = torch.floor((output - 1) / stride) + 1 if stride > 1 else output
        return output.long().clamp(min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.cnn(x)
        x = x.amax(dim=2).permute(0, 2, 1)
        cnn_lengths = self._time_out_lengths(lengths)
        packed = pack_padded_sequence(x, cnn_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _output, (hidden, _cell) = self.lstm(packed)
        state = torch.cat([hidden[-2], hidden[-1]], dim=1)
        embedding = self.embed_proj(self.dropout(state))
        logits = self.classifier(embedding)
        return logits, F.normalize(embedding, p=2, dim=1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _logits, embedding = self.forward(x, lengths)
        return embedding


def make_mmwave_dataloaders(samples: list[dict[str, str]], train_speakers: list[int], val_speakers: list[int], label_map: dict[str, int], config: MmwaveTrainingConfig) -> tuple[DataLoader, DataLoader]:
    train_users = {str(speaker) for speaker in train_speakers}
    val_users = {str(speaker) for speaker in val_speakers}
    train_dataset = MmwaveDataset([sample for sample in samples if sample["user_id"] in train_users], label_map, augment=True)
    val_dataset = MmwaveDataset([sample for sample in samples if sample["user_id"] in val_users], label_map, augment=False)
    if not train_dataset or not val_dataset:
        raise RuntimeError(f"Empty mmWave dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    return (
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_mmwave_batch),
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_mmwave_batch),
    )


@torch.no_grad()
def evaluate_mmwave_model(model: MmwaveCNNLSTMEncoder, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for padded, lengths, labels, _samples in loader:
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
        logits, _embedding = model(padded, lengths)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_mmwave_model(model: MmwaveCNNLSTMEncoder, train_loader: DataLoader, val_loader: DataLoader, config: MmwaveTrainingConfig, checkpoint_path: str | Path, device: torch.device) -> MmwaveCNNLSTMEncoder:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    checkpoint = Path(checkpoint_path)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc, patience_count = 0.0, 0
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for padded, lengths, labels, _samples in train_loader:
            padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
            optimizer.zero_grad()
            logits, _embedding = model(padded, lengths)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
        val_loss, val_acc = evaluate_mmwave_model(model, val_loader, criterion, device)
        scheduler.step()
        print(f"Epoch {epoch:03d}/{config.max_epochs} train_loss={total_loss / max(total, 1):.4f} train_acc={correct / max(total, 1):.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}", flush=True)
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
def extract_mmwave_embeddings(model: MmwaveCNNLSTMEncoder, samples: list[dict[str, str]], label_map: dict[str, int], output_path: str | Path, device: torch.device) -> int:
    model.eval()
    embeddings, labels, user_ids, group_names, sample_names = [], [], [], [], []
    for sample in samples:
        rtm = load_mmwave_rtm(sample["path"])
        if rtm is None:
            continue
        x = torch.from_numpy(rtm).unsqueeze(0).to(device)
        lengths = torch.tensor([len(rtm)], dtype=torch.long, device=device)
        embeddings.append(model.encode(x, lengths).squeeze(0).cpu().numpy())
        labels.append(label_map[sample["label_str"]])
        user_ids.append(sample["user_id"])
        group_names.append(sample["group_name"])
        sample_names.append(sample["sample_name"])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.stack(embeddings).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        user_ids=np.asarray(user_ids),
        group_names=np.asarray(group_names),
        sample_names=np.asarray(sample_names),
    )
    return len(embeddings)


def save_label_map(label_map: dict[str, int], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
