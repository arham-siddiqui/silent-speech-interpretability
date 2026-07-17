"""Utilities for training silent-sensor students against teacher targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent


@dataclass
class StudentTrainConfig:
    batch_size: int = 64
    max_epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    ce_weight: float = 0.2
    patience: int = 8


def target_alignment_loss(predicted: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Mean per-sample squared L2 distance between normalized teacher targets."""
    normalized_teacher = F.normalize(teacher, p=2, dim=-1)
    return ((predicted - normalized_teacher) ** 2).sum(dim=-1).mean()


def make_student_arrays(
    modality_arrays: dict[str, np.ndarray],
    teacher_targets: np.ndarray,
    labels: np.ndarray,
    modalities: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.concatenate([modality_arrays[modality].astype(np.float32) for modality in modalities], axis=1)
    y_teacher = teacher_targets.astype(np.float32)
    y_label = labels.astype(np.int64)
    return x, y_teacher, y_label


def _loader(x: np.ndarray, y_teacher: np.ndarray, y_label: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y_teacher, dtype=torch.float32),
            torch.tensor(y_label, dtype=torch.long),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )


@torch.no_grad()
def evaluate_student(model: ArticulatoryStudent, loader: DataLoader, device: torch.device, ce_weight: float = 0.2) -> dict[str, float]:
    model.eval()
    total, mse_total, ce_total, cosine_total, correct = 0, 0.0, 0.0, 0.0, 0
    for x, teacher, labels in loader:
        x, teacher, labels = x.to(device), teacher.to(device), labels.to(device)
        output = model(x)
        normalized_teacher = F.normalize(teacher, p=2, dim=-1)
        mse = F.mse_loss(output["target"], normalized_teacher, reduction="sum")
        ce = F.cross_entropy(output["logits"], labels, reduction="sum")
        mse_total += float(mse.item())
        ce_total += float(ce.item())
        cosine_total += float(F.cosine_similarity(output["target"], normalized_teacher, dim=-1).sum().item())
        correct += int((output["logits"].argmax(dim=1) == labels).sum().item())
        total += int(len(labels))
    return {
        "loss": (mse_total + ce_weight * ce_total) / max(total, 1),
        "mse": mse_total / max(total, 1),
        "cosine_similarity": cosine_total / max(total, 1),
        "ce": ce_total / max(total, 1),
        "accuracy": correct / max(total, 1),
    }


def train_student(
    model: ArticulatoryStudent,
    train_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    val_arrays: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: StudentTrainConfig,
    device: torch.device,
) -> tuple[ArticulatoryStudent, dict[str, float]]:
    train_loader = _loader(*train_arrays, batch_size=config.batch_size, shuffle=True)
    val_loader = _loader(*val_arrays, batch_size=config.batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_metrics = {"loss": float("inf"), "mse": float("inf"), "ce": float("inf"), "accuracy": 0.0}
    patience_count = 0
    model.to(device)
    for epoch in range(1, config.max_epochs + 1):
        model.train()
        for x, teacher, labels in train_loader:
            x, teacher, labels = x.to(device), teacher.to(device), labels.to(device)
            optimizer.zero_grad()
            output = model(x)
            mse = target_alignment_loss(output["target"], teacher)
            ce = F.cross_entropy(output["logits"], labels)
            loss = mse + config.ce_weight * ce
            loss.backward()
            optimizer.step()

        metrics = evaluate_student(model, val_loader, device, config.ce_weight)
        print(
            f"Epoch {epoch:03d}/{config.max_epochs} val_loss={metrics['loss']:.4f} "
            f"val_mse={metrics['mse']:.4f} val_acc={metrics['accuracy']:.3f}",
            flush=True,
        )
        if metrics["loss"] < best_metrics["loss"]:
            best_metrics = metrics
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config.patience:
                break

    model.load_state_dict(best_state)
    return model, best_metrics
