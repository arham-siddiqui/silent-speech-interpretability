import numpy as np

from silent_speech_interpretability.models.encoders.lip import (
    LipTrainingConfig,
    compute_velocity,
    normalize_landmarks,
)


def test_normalize_landmarks_centers_and_scales():
    landmarks = np.array([[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]], dtype=np.float32)
    normalized = normalize_landmarks(landmarks)
    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.max(np.linalg.norm(normalized, axis=1)) <= 1.0 + 1e-6


def test_compute_velocity_shape():
    sequence = np.random.default_rng(0).normal(size=(12, 40)).astype(np.float32)
    velocity = compute_velocity(sequence)
    assert velocity.shape == sequence.shape


def test_lip_training_config_defaults():
    config = LipTrainingConfig()
    assert config.embedding_dim == 128
    assert config.max_epochs > 0
