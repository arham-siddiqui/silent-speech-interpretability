import numpy as np
import torch

from silent_speech_interpretability.interp.causal_ablation import reconstructed_bottleneck
from silent_speech_interpretability.interp.feature_ranking import eta_squared, feature_rankings
from silent_speech_interpretability.interp.sae import SparseAutoencoder


def test_sparse_autoencoder_shapes_and_decoder_norms():
    model = SparseAutoencoder(input_dim=8, feature_dim=16)
    output = model(torch.randn(4, 8))
    assert output["features"].shape == (4, 16)
    assert output["reconstruction"].shape == (4, 8)
    assert torch.allclose(model.decoder.weight.norm(dim=0), torch.ones(16), atol=1e-5)


def test_feature_ranking_prefers_label_selective_feature():
    labels = np.repeat([0, 1], 10)
    features = np.column_stack([labels, np.ones(20), np.tile([0, 1], 10)]).astype(np.float32)
    rankings = feature_rankings(features, labels, labels, np.tile([0, 1], 10))
    assert eta_squared(features, labels)[0] > 0.99
    assert rankings["rank"][0] == 0


def test_reconstructed_bottleneck_ablation_changes_selected_feature():
    torch.manual_seed(1)
    model = SparseAutoencoder(input_dim=4, feature_dim=6)
    x = np.random.default_rng(1).normal(size=(3, 4)).astype(np.float32)
    _baseline, baseline_features = reconstructed_bottleneck(model, x, np.zeros(4), np.ones(4))
    _ablated, ablated_features = reconstructed_bottleneck(model, x, np.zeros(4), np.ones(4), [1], 0.0)
    assert np.any(baseline_features[:, 1] != 0.0)
    assert np.all(ablated_features[:, 1] == 0.0)
