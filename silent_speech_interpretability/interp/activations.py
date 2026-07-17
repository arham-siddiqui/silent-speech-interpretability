"""Activation extraction helpers."""

from __future__ import annotations

import numpy as np
import torch

from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent


@torch.no_grad()
def extract_student_activations(
    model: ArticulatoryStudent,
    sensor_inputs: np.ndarray,
    batch_size: int = 128,
) -> dict[str, np.ndarray]:
    """Extract named student representations on the model's current device."""
    model.eval()
    device = next(model.parameters()).device
    collected: dict[str, list[np.ndarray]] = {}
    for start in range(0, len(sensor_inputs), batch_size):
        batch = torch.tensor(sensor_inputs[start : start + batch_size], dtype=torch.float32, device=device)
        activations = model.extract_activations(batch)
        for layer, values in activations.items():
            collected.setdefault(layer, []).append(values.detach().cpu().numpy().astype(np.float32))
    return {layer: np.concatenate(chunks, axis=0) for layer, chunks in collected.items()}
