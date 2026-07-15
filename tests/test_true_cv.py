from pathlib import Path

from silent_speech_interpretability.evals.true_cv import (
    configured_fold_embedding_paths,
    expected_fold_embedding_paths,
    missing_embedding_paths,
)


def test_expected_fold_embedding_paths():
    paths = expected_fold_embedding_paths("artifacts/embeddings/speaker_cv", 2, modalities=("lip", "laser"))
    assert paths["lip"] == Path("artifacts/embeddings/speaker_cv/fold_2/lip_embeddings.npz")
    assert paths["laser"] == Path("artifacts/embeddings/speaker_cv/fold_2/laser_embeddings.npz")


def test_configured_fold_embedding_paths_override_defaults(tmp_path):
    custom = tmp_path / "lip_fold0.npz"
    paths = configured_fold_embedding_paths(
        {"fold_embedding_paths": {"fold_0": {"lip": str(custom)}}},
        0,
        modalities=("lip",),
    )
    assert paths["lip"] == custom


def test_missing_embedding_paths_reports_absent_files(tmp_path):
    present = tmp_path / "present.npz"
    present.touch()
    missing = missing_embedding_paths({"lip": present, "laser": tmp_path / "missing.npz"})
    assert missing == {"laser": str(tmp_path / "missing.npz")}
