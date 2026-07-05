from silent_speech_interpretability.data.manifest import build_intersection_manifest
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
