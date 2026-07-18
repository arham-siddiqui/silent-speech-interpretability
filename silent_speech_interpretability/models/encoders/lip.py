"""Lip landmark LSTM encoder and fold-specific embedding extraction."""

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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
from torch.utils.data import DataLoader, Dataset


LIP_START, LIP_END = 48, 68


@dataclass
class LipTrainingConfig:
    hidden_size: int = 256
    num_layers: int = 2
    embedding_dim: int = 128
    dropout: float = 0.3
    batch_size: int = 32
    lr: float = 3e-4
    max_epochs: int = 80
    patience: int = 25
    lambda_supcon: float = 0.5
    lambda_dann: float = 0.3
    supcon_temperature: float = 0.07
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1


class GradRevFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        (alpha,) = ctx.saved_tensors
        return -alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    scale = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    return GradRevFunction.apply(x, scale)


class SupConLoss(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        similarities = torch.mm(features, features.T) / self.temperature
        similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye
        if positive_mask.sum() == 0:
            return torch.zeros((), device=features.device, requires_grad=True)

        exp_similarities = torch.exp(similarities) * (~eye).float()
        log_denominator = torch.log(exp_similarities.sum(dim=1, keepdim=True) + 1e-8)
        log_probability = similarities - log_denominator
        positive_counts = positive_mask.float().sum(dim=1).clamp(min=1)
        per_sample = (log_probability * positive_mask.float()).sum(dim=1) / positive_counts
        return -per_sample.mean()


def natural_sort_key(path: str | Path) -> list[object]:
    name = Path(path).name
    tokens = re.split(r"(\d+)", name)
    return [int(token) if token.isdigit() else token.lower() for token in tokens]


def resolve_kinect_root(rvtall_base: str | Path) -> Path:
    base = Path(rvtall_base).expanduser().resolve()
    if base.name == "kinect_processed":
        return base
    if (base / "kinect_processed").is_dir():
        return base / "kinect_processed"
    raise FileNotFoundError(f"Could not find kinect_processed under {base}")


def list_sorted_npy_files(directory: str | Path) -> list[Path]:
    return sorted(Path(directory).glob("*.npy"), key=natural_sort_key)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    landmarks = np.asarray(landmarks, dtype=np.float32)
    landmarks = landmarks - landmarks.mean(axis=0)
    scale = np.max(np.linalg.norm(landmarks, axis=1)) + 1e-8
    return landmarks / scale


def load_lip_sequence(landmarkers_dir: str | Path) -> np.ndarray | None:
    frames = []
    for path in list_sorted_npy_files(landmarkers_dir):
        arr = np.load(path)
        if arr.ndim != 2 or arr.shape[0] < 68 or arr.shape[1] < 2:
            continue
        lip = normalize_landmarks(arr[LIP_START:LIP_END, :2])
        frames.append(lip.flatten())
    if len(frames) < 5:
        return None
    return np.asarray(frames, dtype=np.float32)


def compute_velocity(sequence: np.ndarray) -> np.ndarray:
    return np.gradient(sequence, axis=0).astype(np.float32)


def build_lip_sample_list(rvtall_base: str | Path) -> list[dict[str, str]]:
    root = resolve_kinect_root(rvtall_base)
    user_dirs = sorted([path for path in root.iterdir() if path.is_dir()], key=natural_sort_key)
    samples = []
    for user_dir in user_dirs:
        user_id = user_dir.name
        for group_prefix in ("sentences", "vowel", "word"):
            for group_dir in sorted(glob.glob(str(user_dir / f"{group_prefix}*")), key=natural_sort_key):
                group_path = Path(group_dir)
                videos_dir = group_path / "videos"
                if not videos_dir.is_dir():
                    continue
                video_pattern = "video_[0-9]*" if user_id == "1" else "video_proc_*"
                for video_dir in sorted(glob.glob(str(videos_dir / video_pattern)), key=natural_sort_key):
                    landmarkers_dir = Path(video_dir) / "landmarkers_cv"
                    if landmarkers_dir.is_dir():
                        samples.append(
                            {
                                "user_id": user_id,
                                "group_name": group_path.name,
                                "video_name": Path(video_dir).name,
                                "landmarkers_dir": str(landmarkers_dir),
                                "label_str": group_path.name,
                            }
                        )
    return samples


class LipDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]], label_map: dict[str, int], speaker_map: dict[str, int], augment: bool = False):
        self.label_map = label_map
        self.speaker_map = speaker_map
        self.augment = augment
        self.items = []
        for sample in samples:
            sequence = load_lip_sequence(sample["landmarkers_dir"])
            if sequence is None:
                continue
            velocity = compute_velocity(sequence)
            features = np.concatenate([sequence, velocity], axis=1)
            self.items.append(
                (
                    features,
                    label_map[sample["label_str"]],
                    speaker_map.get(sample["user_id"], 0),
                    sample["video_name"],
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        sequence, label, speaker, video_name = self.items[idx]
        if self.augment:
            sequence = self._augment(sequence)
        return torch.from_numpy(sequence), label, speaker, video_name

    @staticmethod
    def _augment(sequence: np.ndarray) -> np.ndarray:
        if len(sequence) > 10:
            keep = int(np.random.uniform(0.75, 1.0) * len(sequence))
            start = np.random.randint(0, len(sequence) - keep + 1)
            sequence = sequence[start : start + keep]
        if np.random.rand() < 0.5:
            speed = np.random.uniform(0.85, 1.15)
            new_length = max(5, int(len(sequence) / speed))
            source_idx = np.arange(len(sequence))
            target_idx = np.linspace(0, len(sequence) - 1, new_length)
            sequence = np.stack([np.interp(target_idx, source_idx, sequence[:, dim]) for dim in range(sequence.shape[1])], axis=1)
        noise = np.random.normal(0, 0.005, sequence.shape).astype(np.float32)
        return (sequence + noise).astype(np.float32)


def collate_lip_batch(batch):
    sequences, labels, speakers, video_names = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return (
        padded,
        lengths,
        torch.tensor(labels, dtype=torch.long),
        torch.tensor(speakers, dtype=torch.long),
        list(video_names),
    )


class LipExtractionDataset(Dataset):
    def __init__(self, samples: list[dict[str, str]], label_map: dict[str, int]):
        self.items = []
        for sample in samples:
            sequence = load_lip_sequence(sample["landmarkers_dir"])
            if sequence is None:
                continue
            features = np.concatenate([sequence, compute_velocity(sequence)], axis=1)
            self.items.append((features, label_map[sample["label_str"]], sample))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        features, label, sample = self.items[idx]
        return torch.from_numpy(features), label, sample


def collate_lip_extraction_batch(batch):
    sequences, labels, samples = zip(*batch)
    lengths = torch.tensor([len(sequence) for sequence in sequences], dtype=torch.long)
    padded = pad_sequence(sequences, batch_first=True)
    return padded, lengths, torch.tensor(labels, dtype=torch.long), list(samples)


class LipLSTMV2(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        num_speakers: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attn_proj = nn.Linear(hidden_size * 2, 1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim), nn.LayerNorm(embedding_dim))
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.speaker_head = nn.Sequential(nn.Linear(embedding_dim, 64), nn.ReLU(), nn.Linear(64, max(num_speakers, 1)))

    def _attend(self, lstm_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        scores = self.attn_proj(lstm_out).squeeze(-1)
        max_length = lstm_out.size(1)
        mask = torch.arange(max_length, device=lstm_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)
        return (lstm_out * weights).sum(dim=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, dann_alpha: float = 0.0):
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        pooled = self.dropout(self._attend(lstm_out, lengths))
        raw_embedding = self.embed_proj(pooled)
        class_logits = self.classifier(raw_embedding)
        speaker_logits = self.speaker_head(grad_reverse(raw_embedding, alpha=dann_alpha))
        embedding = F.normalize(raw_embedding, p=2, dim=1)
        return class_logits, speaker_logits, embedding

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _, _, embedding = self.forward(x, lengths, dann_alpha=0.0)
        return embedding

    def encode_sequence(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Project every valid BiLSTM time step into the trained embedding space."""
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        return F.normalize(self.embed_proj(output), p=2, dim=-1), lengths


def dann_alpha(epoch: int, total_epochs: int, max_alpha: float) -> float:
    progress = epoch / total_epochs
    return max_alpha * (2.0 / (1.0 + np.exp(-10.0 * progress)) - 1.0)


def train_lip_model(
    model: LipLSTMV2,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: LipTrainingConfig,
    checkpoint_path: str | Path,
    device: torch.device,
    resume: bool = False,
) -> LipLSTMV2:
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    supcon = SupConLoss(config.supcon_temperature)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5)

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
        print(f"Resuming lip training from epoch {start_epoch} using {state_checkpoint}", flush=True)
    elif resume and checkpoint.exists():
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        print(f"Warm-starting lip training from model checkpoint {checkpoint}", flush=True)

    for epoch in range(start_epoch, config.max_epochs + 1):
        model.train()
        alpha = dann_alpha(epoch, config.max_epochs, config.lambda_dann)
        total_loss = 0.0
        correct = 0
        total = 0
        for padded, lengths, labels, speakers, _video_names in train_loader:
            padded = padded.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)
            speakers = speakers.to(device)
            optimizer.zero_grad()
            class_logits, speaker_logits, embedding = model(padded, lengths, dann_alpha=alpha)
            ce_loss = criterion(class_logits, labels)
            loss = ce_loss + config.lambda_supcon * supcon(embedding, labels) + alpha * F.cross_entropy(speaker_logits, speakers)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += ce_loss.item() * len(labels)
            correct += (class_logits.argmax(dim=1) == labels).sum().item()
            total += len(labels)

        val_loss, val_acc = evaluate_lip_model(model, val_loader, criterion, device)
        scheduler.step(val_acc)
        train_acc = correct / max(total, 1)
        print(
            f"Epoch {epoch:03d}/{config.max_epochs} "
            f"train_loss={total_loss / max(total, 1):.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f} alpha={alpha:.3f}"
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
def evaluate_lip_model(model: LipLSTMV2, loader: DataLoader, criterion: nn.Module, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    for padded, lengths, labels, _speakers, _video_names in loader:
        padded = padded.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)
        class_logits, _speaker_logits, _embedding = model(padded, lengths, dann_alpha=0.0)
        total_loss += criterion(class_logits, labels).item() * len(labels)
        correct += (class_logits.argmax(dim=1) == labels).sum().item()
        total += len(labels)
    return total_loss / max(total, 1), correct / max(total, 1)


@torch.no_grad()
def extract_lip_embeddings(
    model: LipLSTMV2,
    samples: list[dict[str, str]],
    label_map: dict[str, int],
    output_path: str | Path,
    device: torch.device,
    batch_size: int = 64,
) -> int:
    model.eval()
    embeddings, labels, user_ids, group_names, video_names = [], [], [], [], []
    dataset = LipExtractionDataset(samples, label_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_lip_extraction_batch)
    for padded, lengths, batch_labels, batch_samples in loader:
        padded = padded.to(device)
        lengths = lengths.to(device)
        batch_embeddings = model.encode(padded, lengths).cpu().numpy()
        embeddings.extend(batch_embeddings)
        labels.extend(batch_labels.numpy().tolist())
        for sample in batch_samples:
            user_ids.append(sample["user_id"])
            group_names.append(sample["group_name"])
            video_names.append(sample["video_name"])

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        embeddings=np.stack(embeddings).astype(np.float32),
        labels=np.asarray(labels, dtype=np.int32),
        user_ids=np.asarray(user_ids),
        group_names=np.asarray(group_names),
        video_names=np.asarray(video_names),
    )
    return len(embeddings)


def make_lip_dataloaders(
    samples: list[dict[str, str]],
    train_speakers: list[int],
    val_speakers: list[int],
    label_map: dict[str, int],
    config: LipTrainingConfig,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    train_speaker_strings = {str(speaker) for speaker in train_speakers}
    val_speaker_strings = {str(speaker) for speaker in val_speakers}
    train_samples = [sample for sample in samples if sample["user_id"] in train_speaker_strings]
    val_samples = [sample for sample in samples if sample["user_id"] in val_speaker_strings]
    train_users = sorted({sample["user_id"] for sample in train_samples}, key=lambda value: int(value))
    speaker_map = {user_id: idx for idx, user_id in enumerate(train_users)}

    train_dataset = LipDataset(train_samples, label_map, speaker_map, augment=True)
    val_dataset = LipDataset(val_samples, label_map, speaker_map, augment=False)
    if not train_dataset or not val_dataset:
        raise RuntimeError(f"Empty lip dataset: train={len(train_dataset)}, val={len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_lip_batch)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_lip_batch)
    return train_loader, val_loader, speaker_map


def save_label_map(label_map: dict[str, int], path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(label_map, indent=2), encoding="utf-8")
