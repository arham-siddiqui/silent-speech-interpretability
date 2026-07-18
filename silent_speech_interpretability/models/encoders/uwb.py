"""UWB radar CNN+LSTM encoder and embedding extraction."""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset

EXPECTED_RANGE_BINS = 205


@dataclass
class UWBTrainingConfig:
    batch_size: int = 16
    lr: float = 3e-4
    max_epochs: int = 80
    patience: int = 25
    hidden_size: int = 128
    embedding_dim: int = 128
    dropout: float = 0.3
    lambda_dann: float = 0.3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


def natural_sort_key(path: str | Path) -> list[object]:
    name = Path(path).name
    return [int(token) if token.isdigit() else token.lower() for token in re.split(r"(\d+)", name)]


def resolve_uwb_root(rvtall_base: str | Path) -> Path:
    base = Path(rvtall_base).expanduser().resolve()
    if base.name == "uwb_processed":
        return base
    if (base / "uwb_processed").is_dir():
        return base / "uwb_processed"
    raise FileNotFoundError(f"Could not find uwb_processed under {base}")


def normalise_group_name(folder_name: str) -> str:
    return folder_name.replace("_", "", 1)


def load_uwb_rtm(ant1_path: str | Path, ant2_path: str | Path) -> np.ndarray | None:
    rtm1 = np.load(ant1_path).astype(np.float32)
    rtm2 = np.load(ant2_path).astype(np.float32)
    if rtm1.shape[0] != EXPECTED_RANGE_BINS or rtm2.shape[0] != EXPECTED_RANGE_BINS:
        return None
    time_steps = min(rtm1.shape[1], rtm2.shape[1])
    if time_steps < 10:
        return None
    rtm = np.stack([rtm1[:, :time_steps], rtm2[:, :time_steps]], axis=0)
    rtm = rtm - rtm.mean(axis=2, keepdims=True)
    std_per_bin = rtm.std(axis=2, keepdims=True)
    rtm = rtm / np.where(std_per_bin > 1e-8, std_per_bin, 1.0)
    rtm = np.clip(rtm, -3.0, 3.0)
    global_std = rtm.std()
    if global_std > 1e-8:
        rtm = rtm / global_std
    return rtm.transpose(2, 0, 1).astype(np.float32)


def build_uwb_sample_list(rvtall_base: str | Path) -> list[dict[str, str]]:
    root = resolve_uwb_root(rvtall_base)
    samples = []
    for user_dir in sorted([path for path in root.iterdir() if path.is_dir()], key=natural_sort_key):
        user_id = user_dir.name
        for group_dir in sorted([path for path in user_dir.iterdir() if path.is_dir()], key=natural_sort_key):
            group_name = normalise_group_name(group_dir.name)
            if not any(group_name.startswith(prefix) for prefix in ("sentences", "vowel", "word")):
                continue
            pairs: dict[str, dict[str, str]] = defaultdict(dict)
            for npy_path in sorted(group_dir.glob("*.npy"), key=natural_sort_key):
                parts = npy_path.stem.split("_")
                if len(parts) < 5:
                    continue
                antenna, sample_id = parts[-2], parts[-1]
                if antenna in {"1", "2"}:
                    pairs[sample_id][antenna] = str(npy_path)
            for sample_id, antennas in sorted(pairs.items(), key=lambda item: natural_sort_key(item[0])):
                if "1" in antennas and "2" in antennas:
                    samples.append(
                        {
                            "user_id": user_id,
                            "group_name": group_name,
                            "sample_name": sample_id,
                            "ant1_path": antennas["1"],
                            "ant2_path": antennas["2"],
                            "label_str": group_name,
                        }
                    )
    return samples


class UWBDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]], label_map: dict[str, int], speaker_map: dict[str, int], augment: bool = False):
        self.augment = augment
        self.items = []
        for sample in samples:
            try:
                shape1 = np.load(sample["ant1_path"], mmap_mode="r").shape
                shape2 = np.load(sample["ant2_path"], mmap_mode="r").shape
            except Exception:
                continue
            if (
                len(shape1) == 2
                and len(shape2) == 2
                and shape1[0] == EXPECTED_RANGE_BINS
                and shape2[0] == EXPECTED_RANGE_BINS
                and min(shape1[1], shape2[1]) >= 10
            ):
                self.items.append((sample, label_map[sample["label_str"]], speaker_map.get(sample["user_id"], 0)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sample, label, speaker = self.items[idx]
        rtm = load_uwb_rtm(sample["ant1_path"], sample["ant2_path"])
        if rtm is None:
            raise RuntimeError(f"Invalid UWB RTM after validation: {sample['sample_name']}")
        if self.augment:
            if rtm.shape[0] > 20:
                keep = int(np.random.uniform(0.8, 1.0) * rtm.shape[0])
                start = np.random.randint(0, rtm.shape[0] - keep + 1)
                rtm = rtm[start : start + keep]
            rtm = rtm + np.random.normal(0, 0.02, rtm.shape).astype(np.float32)
        return torch.from_numpy(rtm.astype(np.float32)), label, speaker, sample


def collate_uwb_batch(batch):
    rtms, labels, speakers, samples = zip(*batch)
    lengths = torch.tensor([rtm.shape[0] for rtm in rtms], dtype=torch.long)
    padded = pad_sequence(rtms, batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long), torch.tensor(speakers, dtype=torch.long), list(samples)


class GradRevFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    return GradRevFunction.apply(x, torch.tensor(alpha, dtype=x.dtype, device=x.device))


def dann_alpha(epoch: int, total_epochs: int, max_alpha: float) -> float:
    p = epoch / max(total_epochs, 1)
    return float(max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0))


class ResBlock2D(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x + self.net(x))


class UWBEncoderV2(nn.Module):
    RANGE_STRIDES = (2, 2, 2, 1)
    NUM_CONV_LAYERS = 4

    def __init__(self, num_classes: int, num_speakers: int, hidden_size: int = 128, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()

        def conv_block(in_channels: int, out_channels: int, range_stride: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(range_stride, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                ResBlock2D(out_channels),
            )

        self.cnn = nn.Sequential(conv_block(2, 32, 2), conv_block(32, 64, 2), conv_block(64, 128, 2), conv_block(128, 128, 1))
        self.lstm = nn.LSTM(input_size=128, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.attn_proj = nn.Linear(hidden_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim), nn.LayerNorm(embedding_dim))
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.speaker_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, max(num_speakers, 1)))

    def _time_out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        output = lengths.float()
        for _ in range(self.NUM_CONV_LAYERS):
            output = torch.floor((output - 1) / 2) + 1
        return output.long().clamp(min=1)

    def _attend(self, lstm_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        scores = self.attn_proj(lstm_out).squeeze(-1)
        max_t = lstm_out.size(1)
        mask = torch.arange(max_t, device=lstm_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
        weights = F.softmax(scores.masked_fill(mask, float("-inf")), dim=1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, dann_alpha_value: float = 0.0):
        lstm_out, cnn_lengths = self._encode_backbone(x, lengths)
        raw_embedding = self.embed_proj(self.dropout(self._attend(lstm_out, cnn_lengths)))
        class_logits = self.classifier(raw_embedding)
        speaker_logits = self.speaker_head(grad_reverse(raw_embedding, alpha=dann_alpha_value))
        return class_logits, speaker_logits, F.normalize(raw_embedding, p=2, dim=1)

    def _encode_backbone(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = x.permute(0, 2, 3, 1)
        x = self.cnn(x)
        x = x.amax(dim=2).permute(0, 2, 1)
        cnn_lengths = self._time_out_lengths(lengths)
        packed = pack_padded_sequence(x, cnn_lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _hidden = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        return lstm_out, cnn_lengths

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _class_logits, _speaker_logits, embedding = self.forward(x, lengths, dann_alpha_value=0.0)
        return embedding

    def encode_sequence(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output, output_lengths = self._encode_backbone(x, lengths)
        return F.normalize(self.embed_proj(output), p=2, dim=-1), output_lengths


def make_uwb_dataloaders(samples: list[dict[str, str]], train_speakers: list[int], val_speakers: list[int], label_map: dict[str, int], config: UWBTrainingConfig) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_users = {str(speaker) for speaker in train_speakers}
    val_users = {str(speaker) for speaker in val_speakers}
    speaker_map = {user: idx for idx, user in enumerate(sorted(train_users, key=lambda user: int(user)))}
    train_dataset = UWBDataset([sample for sample in samples if sample["user_id"] in train_users], label_map, speaker_map, augment=True)
    val_dataset = UWBDataset([sample for sample in samples if sample["user_id"] in val_users], label_map, speaker_map, augment=False)
    if not train_dataset or not val_dataset:
        raise RuntimeError(f"Empty UWB dataset: train={len(train_dataset)}, val={len(val_dataset)}")
    return (
        DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_uwb_batch),
        DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_uwb_batch),
        speaker_map,
    )


@torch.no_grad()
def evaluate_uwb_model(model: UWBEncoderV2, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for padded, lengths, labels, _speakers, _samples in loader:
        padded, lengths, labels = padded.to(device), lengths.to(device), labels.to(device)
        logits, _speaker_logits, _embedding = model(padded, lengths, dann_alpha_value=0.0)
        total_loss += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_uwb_model(model: UWBEncoderV2, train_loader: DataLoader, val_loader: DataLoader, config: UWBTrainingConfig, checkpoint_path: str | Path, device: torch.device) -> UWBEncoderV2:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5)
    checkpoint = Path(checkpoint_path)
    state_checkpoint = checkpoint.with_suffix(".training_state.pt")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_val_acc, patience_count = 0.0, 0
    start_epoch = 1
    if state_checkpoint.exists():
        state = torch.load(state_checkpoint, map_location=device)
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        scheduler.load_state_dict(state["scheduler_state"])
        best_val_acc = float(state.get("best_val_acc", 0.0))
        patience_count = int(state.get("patience_count", 0))
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"Resuming UWB training from epoch {start_epoch} using {state_checkpoint}", flush=True)
    elif checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Warm-starting UWB training from model checkpoint {checkpoint}", flush=True)

    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        alpha = dann_alpha(epoch, config.max_epochs, config.lambda_dann)
        total_loss, correct, total = 0.0, 0, 0
        for padded, lengths, labels, speakers, _samples in train_loader:
            padded, lengths, labels, speakers = padded.to(device), lengths.to(device), labels.to(device), speakers.to(device)
            optimizer.zero_grad()
            class_logits, speaker_logits, _embedding = model(padded, lengths, dann_alpha_value=alpha)
            ce_loss = criterion(class_logits, labels)
            loss = ce_loss + alpha * F.cross_entropy(speaker_logits, speakers)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += ce_loss.item() * len(labels)
            correct += (class_logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)
        val_loss, val_acc = evaluate_uwb_model(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        print(f"Epoch {epoch:03d}/{config.max_epochs} train_loss={total_loss / max(total, 1):.4f} train_acc={correct / max(total, 1):.3f} val_loss={val_loss:.4f} val_acc={val_acc:.3f} dann_alpha={alpha:.3f}", flush=True)
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
    if checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model


@torch.no_grad()
def extract_uwb_embeddings(model: UWBEncoderV2, samples: list[dict[str, str]], label_map: dict[str, int], output_path: str | Path, device: torch.device) -> int:
    model.eval()
    embeddings, labels, user_ids, group_names, sample_names = [], [], [], [], []
    for sample in samples:
        rtm = load_uwb_rtm(sample["ant1_path"], sample["ant2_path"])
        if rtm is None:
            continue
        x = torch.from_numpy(rtm).unsqueeze(0).to(device)
        lengths = torch.tensor([rtm.shape[0]], dtype=torch.long, device=device)
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
