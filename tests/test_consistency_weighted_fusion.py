import numpy as np

from silent_speech_interpretability.models.fusion import borda_count_fusion, consistency_weighted_fusion, equal_weight_fusion


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
