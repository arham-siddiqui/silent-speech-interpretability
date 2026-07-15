import numpy as np

from silent_speech_interpretability.data.manifest import (
    build_alignment_audit,
    build_intersection_manifest,
    resolve_embedding_paths,
)
from silent_speech_interpretability.data.synthetic import make_synthetic_manifest


def test_synthetic_manifest_has_required_columns():
    manifest = make_synthetic_manifest(num_speakers=2, classes_per_speaker=3)
    required = {
        "sample_id",
        "user_id",
        "class_id",
        "class_name",
        "utterance_type",
        "lip_path",
        "mouth_video_path",
        "uwb_path",
        "mmwave_path",
        "laser_path",
        "audio_path",
        "group_name",
    }
    assert required.issubset(manifest.columns)


def test_empty_modality_paths_do_not_enter_intersection():
    manifest = make_synthetic_manifest(num_speakers=1, classes_per_speaker=2)
    assert build_intersection_manifest(manifest).empty


def test_explicit_embedding_path_config_takes_precedence(tmp_path):
    configured = tmp_path / "configured_lip.npz"
    discovered = tmp_path / "lip_embeddings_discovered.npz"
    for path in (configured, discovered):
        np.savez(
            path,
            embeddings=np.zeros((1, 4), dtype=np.float32),
            labels=np.array([0]),
            user_ids=np.array([1]),
            group_names=np.array(["sample_0"]),
        )

    paths, sources = resolve_embedding_paths(
        {"embeddings_dir": str(tmp_path), "embedding_paths": {"lip": str(configured)}},
        [tmp_path],
        synthetic_if_missing=False,
    )

    assert paths["lip"] == configured
    assert sources["lip"] == "configured"


def test_alignment_audit_reports_label_mismatches(tmp_path):
    lip_path = tmp_path / "lip_embeddings.npz"
    laser_path = tmp_path / "laser_embeddings.npz"
    kwargs = {
        "embeddings": np.zeros((2, 4), dtype=np.float32),
        "user_ids": np.array([1, 2]),
        "group_names": np.array(["a", "b"]),
    }
    np.savez(lip_path, labels=np.array([0, 1]), **kwargs)
    np.savez(laser_path, labels=np.array([0, 2]), **kwargs)

    audit = build_alignment_audit({"lip": lip_path, "laser": laser_path})

    assert audit["union_group_count"] == 2
    assert audit["strict_intersection_group_count"] == 2
    assert audit["label_mismatch_count"] == 1
