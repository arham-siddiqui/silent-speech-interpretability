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


class ModalityAttentionTemporalStudent(nn.Module):
    """Separate sensor projections with temporal attention and segment-level fusion gates."""

    def __init__(
        self,
        input_dim: int,
        target_dim: int = 768,
        hidden_dim: int = 256,
        bottleneck_dim: int = 64,
        num_classes: int = 30,
        num_segments: int = 4,
        num_modalities: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        if input_dim % num_modalities != 0:
            raise ValueError("input_dim must be divisible by num_modalities")
        self.num_segments = num_segments
        self.num_modalities = num_modalities
        self.modality_dim = input_dim // num_modalities
        self.modality_norms = nn.ModuleList([nn.LayerNorm(self.modality_dim) for _ in range(num_modalities)])
        self.modality_projections = nn.ModuleList(
            [nn.Linear(self.modality_dim, hidden_dim) for _ in range(num_modalities)]
        )
        self.temporal_scores = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_modalities)])
        self.modality_scores = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(num_modalities)])
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
            nn.LayerNorm(num_modalities * hidden_dim),
            nn.Linear(num_modalities * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        nn.init.normal_(self.position, std=0.02)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        if inputs.ndim != 3 or inputs.shape[1] != self.num_segments:
            raise ValueError(f"inputs must have shape [batch, {self.num_segments}, features]")
        chunks = inputs.split(self.modality_dim, dim=-1)
        states = torch.stack(
            [projection(norm(chunk)) for chunk, norm, projection in zip(
                chunks, self.modality_norms, self.modality_projections, strict=True
            )],
            dim=2,
        )
        ordered_states = states + self.position.unsqueeze(0).unsqueeze(2)
        modality_logits = torch.cat(
            [score(ordered_states[:, :, index]) for index, score in enumerate(self.modality_scores)],
            dim=-1,
        )
        modality_weights = F.softmax(modality_logits, dim=-1)
        fused = (ordered_states * modality_weights.unsqueeze(-1)).sum(dim=2)
        bottleneck = self.temporal_bottleneck(fused)
        target = F.normalize(self.target_head(bottleneck), p=2, dim=-1)

        pooled_modalities = []
        temporal_weights = []
        for index, score in enumerate(self.temporal_scores):
            weights = F.softmax(score(ordered_states[:, :, index]).squeeze(-1), dim=1)
            pooled_modalities.append((states[:, :, index] * weights.unsqueeze(-1)).sum(dim=1))
            temporal_weights.append(weights)
        utterance = torch.cat(pooled_modalities, dim=-1)
        logits = self.classifier(utterance)
        return {
            "bottleneck": bottleneck,
            "target": target,
            "logits": logits,
            "utterance": utterance,
            "temporal_attention": torch.stack(temporal_weights, dim=1),
            "modality_weights": modality_weights,
        }
