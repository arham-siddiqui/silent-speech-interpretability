import numpy as np
import torch

from silent_speech_interpretability.interp.activations import extract_student_activations
from silent_speech_interpretability.interp.probes import content_heldout_indices, fit_linear_probe
from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent


def test_student_activation_shapes():
    model = ArticulatoryStudent(["lip", "laser"], embedding_dim=8, hidden_dim=12, bottleneck_dim=6, target_dim=5)
    activations = extract_student_activations(model, np.zeros((4, 16), dtype=np.float32), batch_size=2)
    assert {key: value.shape for key, value in activations.items()} == {
        "sensor_input": (4, 16),
        "hidden": (4, 12),
        "bottleneck": (4, 6),
        "predicted_hubert": (4, 5),
    }


def test_linear_probe_and_content_split():
    rng = np.random.default_rng(2)
    labels = np.repeat(np.arange(3), 20)
    features = np.eye(3, dtype=np.float32)[labels] + rng.normal(scale=0.01, size=(60, 3))
    result = fit_linear_probe(features[:30], labels[:30], features[30:45], labels[30:45], features[45:], labels[45:])
    assert result["accuracy"] >= 0.9

    train, val, test = content_heldout_indices(np.tile(np.arange(10), 3), seed=3)
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)


def test_extract_activations_matches_forward():
    model = ArticulatoryStudent(["lip"], embedding_dim=4, hidden_dim=8, bottleneck_dim=3, target_dim=5)
    model.eval()
    x = torch.randn(2, 4)
    assert torch.allclose(model(x)["target"], model.extract_activations(x)["predicted_hubert"])
