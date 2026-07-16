"""Laser speckle CNN+LSTM encoder and fold-specific embedding extraction."""

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


@dataclass
class LaserTrainingConfig:
    cnn_channels: tuple[int, ...] = (32, 64, 128)
    cnn_kernels: tuple[int, ...] = (15, 9, 7)
    cnn_strides: tuple[int, ...] = (4, 3, 3)
    hidden_size: int = 128
    num_layers: int = 2
    embedding_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 32
    lr: float = 3e-4
    max_epochs: int = 60
    patience: int = 20
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


def natural_sort_key(path: str | Path) -> list[object]:
    name = Path(path).name
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def resolve_laser_root(rvtall_base: str | Path) -> Path:
    base = Path(rvtall_base).expanduser().resolve()
    if base.name == "laser_processed":
        return base
    if (base / "laser_processed").is_dir():
        return base / "laser_processed"
    raise FileNotFoundError(f"Could not find laser_processed under {base}")


def load_laser_signal(path: str | Path) -> np.ndarray:
    signal = np.load(path).astype(np.float32).flatten()
    std = signal.std()
    if std > 1e-8:
        signal = (signal - signal.mean()) / std
    else:
        signal = signal - signal.mean()
    return signal.astype(np.float32)


def build_laser_sample_list(rvtall_base: str | Path) -> list[dict[str, str]]:
    root = resolve_laser_root(rvtall_base)
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


class LaserDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]], label_map: dict[str, int], augment: bool = False):
        self.augment = augment
        self.items = []
        for sample in samples:
            signal = load_laser_signal(sample["path"])
            if len(signal) < 100:
                continue
            self.items.append((signal, label_map[sample["label_str"]], sample))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        signal, label, sample = self.items[idx]
        if self.augment:
            signal = self._augment(signal)
        return torch.from_numpy(signal.astype(np.float32)), label, sample

    @staticmethod
    def _augment(signal: np.ndarray) -> np.ndarray:
        if len(signal) > 200:
            keep = int(np.random.uniform(0.8, 1.0) * len(signal))
            start = np.random.randint(0, len(signal) - keep + 1)
            signal = signal[start : start + keep]
        signal = signal + np.random.normal(0, 0.01, signal.shape).astype(np.float32)
        signal = signal * float(np.random.uniform(0.9, 1.1))
        return signal.astype(np.float32)


def collate_laser_batch(batch):
    signals, labels, samples = zip(*batch)
    lengths = torch.tensor([len(signal) for signal in signals], dtype=torch.long)
    padded = pad_sequence(signals, batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long), list(samples)


class LaserCNNLSTMEncoder(nn.Module):
    def __init__(
        self,
        num_classes: int,
        cnn_channels: tuple[int, ...] = (32, 64, 128),
        cnn_kernels: tuple[int, ...] = (15, 9, 7),
        cnn_strides: tuple[int, ...] = (4, 3, 3),
        hidden_size: int = 128,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self._cnn_kernels = tuple(cnn_kernels)
        self._cnn_strides = tuple(cnn_strides)
        self._cnn_paddings = tuple(kernel // 2 for kernel in cnn_kernels)

        layers = []
        in_channels = 1
        for out_channels, kernel, stride, padding in zip(cnn_channels, cnn_kernels, cnn_strides, self._cnn_paddings):
            layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(inplace=True),
                ]
            )
            in_channels = out_channels
        self.cnn = nn.Sequential(*layers)
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim), nn.LayerNorm(embedding_dim))
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _cnn_out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        output = lengths.float()
        for kernel, stride, padding in zip(self._cnn_kernels, self._cnn_strides, self._cnn_paddings):
            output = torch.floor((output + 2 * padding - kernel) / stride + 1)
        return output.long().clamp(min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        x = x.unsqueeze(1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)
        cnn_lengths = self._cnn_out_lengths(lengths)
        packed = pack_padded_sequence(x, cnn_lengths.cpu(), batch_first=True, enforce_sorted=False)
        _output, (hidden, _cell) = self.lstm(packed)
        state = torch.cat([hidden[-2], hidden[-1]], dim=1)
        embedding = self.embed_proj(self.dropout(state))
        logits = self.classifier(embedding)
        return logits, F.normalize(embedding, p=2, dim=1)

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _logits, embedding = self.forward(x, lengths)
        return embedding


def make_laser_dataloaders(
    samples: list[dict[str, str]],
    train_speakers: list[int],
    val_speakers: list[int],
    label_map: dict[str, int],
    config: LaserTrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    train_speaker_strings = {str(speaker) for speaker in train_speakers}
    val_speaker_strings = {str(speaker) for speaker in val_speakers}
    train_samples = [sample for sample in samples if sample["user_id"] in train_speaker_strings]
    val_samples = [sample for sample in samples if sample["user_id"] in val_speaker_strings]
    train_dataset = LaserDataset(train_samples, label_map, augment=True)
    val_dataset = LaserDataset(val_samples, label_map, augment=False)
    if not train_dataset or not val_dataset:
        raise RuntimeError(f"Empty laser dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_laser_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_laser_batch)
    return train_loader, val_loader


@torch.no_grad()
def evaluate_laser_model(model: LaserCNNLSTMEncoder, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for padded, lengths, labels, _samples in loader:
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
        logits, _embedding = model(padded, lengths)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_laser_model(
    model: LaserCNNLSTMEncoder,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LaserTrainingConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    resume: bool = False,
) -> LaserCNNLSTMEncoder:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs)
    best_val_acc = 0.0
    patience_count = 0
    checkpoint = Path(checkpoint_path)
    state_checkpoint = checkpoint.with_suffix(".training_state.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    start_epoch = 1

    if resume and state_checkpoint.exists():
        state = torch.load(state_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best_val_acc = float(state.get("best_val_acc", 0.0))
        patience_count = int(state.get("patience_count", 0))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resuming laser training from epoch {start_epoch} using {state_checkpoint}", flush=True)
    elif resume and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Warm-starting laser training from model checkpoint {checkpoint}", flush=True)

    for epoch in range(start_epoch, config.max_epochs + 1):
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
        scheduler.step()
        val_loss, val_acc = evaluate_laser_model(model, val_loader, criterion, device)
        train_acc = correct / max(total, 1)
        print(
            f"Epoch {epoch:03d}/{config.max_epochs} "
            f"train_loss={total_loss / max(total, 1):.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}",
            flush=True,
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_count = 0
            torch.save(model.state_dict(), checkpoint)
        else:
            patience_count += 1
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_acc": best_val_acc,
                "patience_count": patience_count,
                "max_epochs": config.max_epochs,
            },
            state_checkpoint,
        )
        if patience_count >= config.patience:
            break

    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


@torch.no_grad()
def extract_laser_embeddings(
    model: LaserCNNLSTMEncoder,
    samples: list[dict[str, str]],
    label_map: dict[str, int],
    output_path: str | Path,
    device: torch.device,
    batch_size: int = 64,
) -> int:
    model.eval()
    dataset = LaserDataset(samples, label_map, augment=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_laser_batch)
    embeddings, labels, user_ids, group_names, sample_names = [], [], [], [], []
    for padded, lengths, batch_labels, batch_samples in loader:
        padded, lengths = padded.to(device), lengths.to(device)
        batch_embeddings = model.encode(padded, lengths).cpu().numpy()
        embeddings.extend(batch_embeddings)
        labels.extend(batch_labels.numpy().tolist())
        for sample in batch_samples:
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
