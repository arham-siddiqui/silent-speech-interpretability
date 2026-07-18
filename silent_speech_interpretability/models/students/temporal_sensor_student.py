"""Student that maps temporal silent-sensor activations to temporal speech targets."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalSensorStudent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        target_dim: int = 768,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        num_classes: int = 30,
        num_segments: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(num_segments, hidden_dim))
        self.bottleneck = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.target_head = nn.Linear(bottleneck_dim, target_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.num_segments:
            raise ValueError(f"inputs must have shape [batch, {self.num_segments}, features]")
        hidden = self.input_proj(self.input_norm(inputs)) + self.position.unsqueeze(0)
        bottleneck = self.bottleneck(hidden)
        target = F.normalize(self.target_head(bottleneck), p=2, dim=-1)
        logits = self.classifier(bottleneck.mean(dim=1))
        return {"bottleneck": bottleneck, "target": target, "logits": logits}


class MultitaskTemporalSensorStudent(nn.Module):
    """Shared temporal stem with separate ordered-target and utterance heads."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int = 768,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        num_classes: int = 30,
        num_segments: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_segments = num_segments
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.zeros(num_segments, hidden_dim))
        self.temporal_bottleneck = nn.Sequential(
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
            nn.GELU(),
        )
        self.target_head = nn.Linear(bottleneck_dim, target_dim)
        self.classifier = nn.Sequential(
            nn.LayerNorm(num_segments * hidden_dim),
            nn.Linear(num_segments * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.num_segments:
            raise ValueError(f"inputs must have shape [batch, {self.num_segments}, features]")
        shared = self.input_proj(self.input_norm(inputs))
        ordered = shared + self.position.unsqueeze(0)
        bottleneck = self.temporal_bottleneck(ordered)
        target = F.normalize(self.target_head(bottleneck), p=2, dim=-1)
        utterance = shared.flatten(start_dim=1)
        logits = self.classifier(utterance)
        return {"bottleneck": bottleneck, "target": target, "logits": logits, "utterance": utterance}
