"""Sparse autoencoders for student bottleneck feature discovery."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    def __init__(self, input_dim: int, feature_dim: int, top_k: int | None = None):
        super().__init__()
        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.top_k = top_k
        self.encoder = nn.Linear(input_dim, feature_dim)
        self.decoder = nn.Linear(feature_dim, input_dim)
        nn.init.kaiming_uniform_(self.encoder.weight)
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.T)
            self.normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        features = F.relu(self.encoder(x))
        if self.top_k is not None and self.top_k < self.feature_dim:
            values, indices = torch.topk(features, self.top_k, dim=-1)
            sparse = torch.zeros_like(features)
            features = sparse.scatter(-1, indices, values)
        return features

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(features)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.encode(x)
        return {"features": features, "reconstruction": self.decode(features)}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))


@dataclass
class SAETrainConfig:
    feature_dim: int = 512
    l1_coeff: float = 1e-4
    top_k: int | None = 32
    steps: int = 5000
    batch_size: int = 128
    lr: float = 3e-4
    eval_interval: int = 100
    patience_evals: int = 10
    seed: int = 42


def normalized_arrays(train_x: np.ndarray, *arrays: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    mean = train_x.mean(axis=0).astype(np.float32)
    std = train_x.std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)
    normalized = [((array - mean) / std).astype(np.float32) for array in (train_x, *arrays)]
    return mean, std, normalized


@torch.no_grad()
def sae_metrics(model: SparseAutoencoder, x: np.ndarray, device: torch.device) -> dict[str, float]:
    values = torch.tensor(x, dtype=torch.float32, device=device)
    output = model(values)
    residual = output["reconstruction"] - values
    mse = float(residual.square().mean().item())
    variance = float(values.var(unbiased=False).item())
    active = output["features"] > 1e-6
    return {
        "mse": mse,
        "explained_variance": 1.0 - mse / max(variance, 1e-8),
        "mean_active_features": float(active.float().sum(dim=1).mean().item()),
        "feature_density": float(active.float().mean().item()),
        "dead_feature_fraction": float((active.sum(dim=0) == 0).float().mean().item()),
        "mean_feature_activation": float(output["features"].mean().item()),
    }


def train_sae(
    train_x: np.ndarray,
    val_x: np.ndarray,
    config: SAETrainConfig,
    device: torch.device,
) -> tuple[SparseAutoencoder, dict[str, float], np.ndarray, np.ndarray]:
    mean, std, (train_normalized, val_normalized) = normalized_arrays(train_x, val_x)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)
    model = SparseAutoencoder(train_x.shape[1], config.feature_dim, top_k=config.top_k).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    train_tensor = torch.tensor(train_normalized, dtype=torch.float32, device=device)
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    best_score = float("inf")
    patience = 0
    steps_run = 0

    for step in range(1, config.steps + 1):
        indices = rng.integers(0, len(train_tensor), min(config.batch_size, len(train_tensor)))
        batch = train_tensor[torch.tensor(indices, device=device)]
        output = model(batch)
        reconstruction_loss = F.mse_loss(output["reconstruction"], batch)
        sparsity_loss = output["features"].mean()
        loss = reconstruction_loss + config.l1_coeff * sparsity_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        model.normalize_decoder()
        steps_run = step

        if step % config.eval_interval == 0 or step == config.steps:
            metrics = sae_metrics(model, val_normalized, device)
            score = metrics["mse"] + config.l1_coeff * metrics["mean_feature_activation"]
            if score < best_score - 1e-7:
                best_score = score
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= config.patience_evals:
                    break

    model.load_state_dict(best_state)
    model.to(device)
    metrics = sae_metrics(model, val_normalized, device)
    metrics["steps_run"] = float(steps_run)
    return model, metrics, mean, std


def save_sae(
    path: str | Path,
    model: SparseAutoencoder,
    mean: np.ndarray,
    std: np.ndarray,
    config: SAETrainConfig,
    metrics: dict[str, object],
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": model.input_dim,
            "feature_dim": model.feature_dim,
            "top_k": model.top_k,
            "input_mean": np.asarray(mean, dtype=np.float32),
            "input_std": np.asarray(std, dtype=np.float32),
            "config": asdict(config),
            "metrics": metrics,
        },
        output,
    )
    return output


def load_sae(path: str | Path, device: torch.device | str = "cpu") -> tuple[SparseAutoencoder, dict[str, object]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = SparseAutoencoder(int(payload["input_dim"]), int(payload["feature_dim"]), top_k=payload.get("top_k"))
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload


@torch.no_grad()
def encode_sae(model: SparseAutoencoder, x: np.ndarray, mean: np.ndarray, std: np.ndarray, device: torch.device) -> np.ndarray:
    normalized = ((x - mean) / std).astype(np.float32)
    values = torch.tensor(normalized, dtype=torch.float32, device=device)
    return model.encode(values).cpu().numpy().astype(np.float32)
