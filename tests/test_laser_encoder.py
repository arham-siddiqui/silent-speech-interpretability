import numpy as np

from silent_speech_interpretability.models.encoders.laser import LaserTrainingConfig, load_laser_signal


def test_load_laser_signal_zscores(tmp_path):
    path = tmp_path / "sample.npy"
    np.save(path, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    signal = load_laser_signal(path)
    assert signal.dtype == np.float32
    assert abs(float(signal.mean())) < 1e-6
    assert abs(float(signal.std()) - 1.0) < 1e-6


def test_laser_training_config_defaults():
    config = LaserTrainingConfig()
    assert config.embedding_dim == 128
    assert config.cnn_channels[-1] == 128
