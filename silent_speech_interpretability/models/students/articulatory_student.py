"""Student model for distilling teacher speech targets into silent embeddings."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArticulatoryStudent(nn.Module):
    """Predict teacher targets from one or more silent-sensor embeddings."""

    def __init__(
        self,
        modalities: list[str],
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        target_dim: int = 64,
        num_classes: int = 30,
        dropout: float = 0.1,
    ):
        super().__init__()
        if not modalities:
            raise ValueError("ArticulatoryStudent requires at least one modality.")
        self.modalities = list(modalities)
        self.embedding_dim = embedding_dim
        self.target_dim = target_dim
        input_dim = embedding_dim * len(self.modalities)
        self.backbone = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.target_head = nn.Linear(bottleneck_dim, target_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)

    def forward(self, embeddings: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        if isinstance(embeddings, dict):
            x = torch.cat([embeddings[modality] for modality in self.modalities], dim=-1)
        else:
            x = embeddings
        activations = self.extract_activations(x)
        bottleneck = activations["bottleneck"]
        target = self.target_head(bottleneck)
        return {
            "bottleneck": bottleneck,
            "target": F.normalize(target, p=2, dim=-1),
            "logits": self.classifier(bottleneck),
        }

    def extract_activations(self, embeddings: dict[str, torch.Tensor] | torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named student layers without changing checkpoint structure."""
        if isinstance(embeddings, dict):
            sensor_input = torch.cat([embeddings[modality] for modality in self.modalities], dim=-1)
        else:
            sensor_input = embeddings
        normalized_input = self.backbone[0](sensor_input)
        hidden = self.backbone[2](self.backbone[1](normalized_input))
        hidden_for_bottleneck = self.backbone[3](hidden)
        bottleneck = self.backbone[6](self.backbone[5](self.backbone[4](hidden_for_bottleneck)))
        predicted_hubert = F.normalize(self.target_head(bottleneck), p=2, dim=-1)
        return {
            "sensor_input": sensor_input,
            "hidden": hidden,
            "bottleneck": bottleneck,
            "predicted_hubert": predicted_hubert,
        }
