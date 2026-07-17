"""Small bottleneck modules used by student models."""

from __future__ import annotations

import torch
import torch.nn as nn


class BottleneckMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, bottleneck_dim: int = 64, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.LayerNorm(bottleneck_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
