"""Base encoder interfaces used by future modality-specific encoders."""

from __future__ import annotations

import torch
from torch import nn


class EmbeddingEncoder(nn.Module):
    embedding_dim: int

    def forward(self, *args, **kwargs) -> torch.Tensor:  # pragma: no cover - interface
        raise NotImplementedError
