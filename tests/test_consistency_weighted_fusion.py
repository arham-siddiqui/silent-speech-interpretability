import numpy as np

from silent_speech_interpretability.models.fusion import (
    borda_count_fusion,
    consistency_weighted_fusion,
    equal_weight_fusion,
    static_weight_fusion,
)


def test_fusion_shapes_and_weight_normalization():
    probs = {
        "lip": np.array([[0.8, 0.2], [0.4, 0.6]]),
        "laser": np.array([[0.7, 0.3], [0.3, 0.7]]),
        "uwb": np.array([[0.2, 0.8], [0.45, 0.55]]),
    }
    assert equal_weight_fusion(probs).shape == (2, 2)
    assert borda_count_fusion(probs).shape == (2, 2)
    fused, weights = consistency_weighted_fusion(probs)
    assert fused.shape == (2, 2)
    assert weights.shape == (2, 3)
    assert np.allclose(weights.sum(axis=1), 1.0)


def test_static_weight_fusion_uses_modality_weights():
    probs = {
        "lip": np.array([[0.9, 0.1]]),
        "laser": np.array([[0.1, 0.9]]),
    }
    fused = static_weight_fusion(probs, {"lip": 3.0, "laser": 1.0})
    assert fused.shape == (1, 2)
    assert fused[0, 0] > fused[0, 1]
