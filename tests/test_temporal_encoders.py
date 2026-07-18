import torch

from silent_speech_interpretability.models.encoders.laser import LaserCNNLSTMEncoder
from silent_speech_interpretability.models.encoders.lip import LipLSTMV2
from silent_speech_interpretability.models.encoders.mmwave import MmwaveCNNLSTMEncoder
from silent_speech_interpretability.models.encoders.uwb import UWBEncoderV2


def _assert_sequence_output(values: torch.Tensor, lengths: torch.Tensor, batch_size: int, embedding_dim: int) -> None:
    assert values.shape[0] == batch_size
    assert values.shape[2] == embedding_dim
    assert values.shape[1] == int(lengths.max())
    assert torch.all(lengths > 0)
    for index, length in enumerate(lengths):
        norms = values[index, : int(length)].norm(dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5)


def test_lip_encoder_exposes_temporal_embeddings():
    model = LipLSTMV2(
        input_size=80,
        num_classes=3,
        num_speakers=2,
        hidden_size=8,
        num_layers=1,
        embedding_dim=6,
        dropout=0.0,
    ).eval()
    values, lengths = model.encode_sequence(torch.randn(2, 12, 80), torch.tensor([12, 9]))
    _assert_sequence_output(values, lengths, batch_size=2, embedding_dim=6)
    assert lengths.tolist() == [12, 9]


def test_laser_encoder_exposes_downsampled_temporal_embeddings():
    model = LaserCNNLSTMEncoder(
        num_classes=3,
        cnn_channels=(4, 6),
        cnn_kernels=(5, 3),
        cnn_strides=(2, 2),
        hidden_size=5,
        num_layers=1,
        embedding_dim=7,
        dropout=0.0,
    ).eval()
    values, lengths = model.encode_sequence(torch.randn(2, 40), torch.tensor([40, 31]))
    _assert_sequence_output(values, lengths, batch_size=2, embedding_dim=7)
    assert lengths.tolist() == [10, 8]


def test_mmwave_encoder_exposes_downsampled_temporal_embeddings():
    model = MmwaveCNNLSTMEncoder(
        num_classes=3,
        hidden_size=4,
        num_layers=1,
        embedding_dim=5,
        dropout=0.0,
    ).eval()
    values, lengths = model.encode_sequence(torch.randn(2, 24, 513), torch.tensor([24, 17]))
    _assert_sequence_output(values, lengths, batch_size=2, embedding_dim=5)
    assert lengths.tolist() == [3, 3]


def test_uwb_encoder_exposes_downsampled_temporal_embeddings():
    model = UWBEncoderV2(
        num_classes=3,
        num_speakers=2,
        hidden_size=4,
        embedding_dim=5,
        dropout=0.0,
    ).eval()
    values, lengths = model.encode_sequence(torch.randn(2, 32, 2, 205), torch.tensor([32, 21]))
    _assert_sequence_output(values, lengths, batch_size=2, embedding_dim=5)
    assert lengths.tolist() == [2, 2]
