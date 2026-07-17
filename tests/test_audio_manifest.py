from pathlib import Path

import numpy as np
import pandas as pd

from silent_speech_interpretability.data.audio_manifest import attach_rvtall_audio_paths


def _audio(root: Path, user: str, group: str, index: int) -> Path:
    path = root / "kinect_processed" / user / group / "audios" / f"audio_proc_{index}.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    return path.resolve()


def test_audio_manifest_uses_reference_repetition_and_reports_missing(tmp_path: Path):
    expected = _audio(tmp_path, "1", "word1", 2)
    _audio(tmp_path, "1", "word1", 0)
    fallback = _audio(tmp_path, "1", "word2", 3)
    reference = tmp_path / "lip_embeddings.npz"
    np.savez(
        reference,
        embeddings=np.zeros((2, 4), dtype=np.float32),
        labels=np.array([0, 0]),
        user_ids=np.array([1, 1]),
        group_names=np.array(["word1", "word1"]),
        video_names=np.array(["video_0", "video_2"]),
    )
    manifest = pd.DataFrame(
        {
            "user_id": [1, 1, 2],
            "group_name": ["word1", "word2", "word3"],
            "audio_path": ["", "", ""],
        }
    )

    aligned, audit = attach_rvtall_audio_paths(manifest, tmp_path, reference)

    assert aligned["audio_path"].tolist() == [str(expected), str(fallback), ""]
    assert audit["audio_files_discovered"] == 3
    assert audit["exact_reference_matches"] == 1
    assert audit["fallback_latest_matches"] == 1
    assert audit["missing_pairs"] == ["2::word3"]
