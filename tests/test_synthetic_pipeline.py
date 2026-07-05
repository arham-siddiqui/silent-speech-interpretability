from pathlib import Path

from silent_speech_interpretability.data.datasets import EmbeddingDataset
from silent_speech_interpretability.data.synthetic import make_synthetic_embeddings, make_synthetic_manifest


def test_embedding_dataset_aligns_synthetic_modalities(tmp_path: Path):
    manifest = make_synthetic_manifest(num_speakers=2, classes_per_speaker=3)
    paths = make_synthetic_embeddings(tmp_path, manifest, modalities=("lip", "laser"), embedding_dim=8)
    dataset = EmbeddingDataset(paths, modalities=["lip", "laser"])
    item = dataset[0]
    assert len(dataset) == len(manifest)
    assert set(item["embeddings"]) == {"lip", "laser"}
    assert item["embeddings"]["lip"].shape[0] == 8
