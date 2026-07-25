from __future__ import annotations

from pathlib import Path
import struct

import numpy as np

from silent_speech_interpretability.data.external_radar import (
    MAX_FREQUENCY_POINTS,
    discover_external_radar_corpus,
    make_external_split_specs,
    radar_segment_features,
    read_radar_binary,
    select_radar_feature_set,
)


def _write_radar_fixture(path: Path, *, frames: int = 8, steps: int = 4, spectra: int = 8) -> None:
    timestamp = b"2021-01-01"
    parts = [
        struct.pack("<I", len(timestamp)),
        timestamp,
        struct.pack("<B", spectra),
        bytes(range(spectra)),
        struct.pack("<IIII", steps, frames, 0, frames - 1),
    ]
    for offset in range(4):
        parts.append((np.arange(steps, dtype="<i4") + offset).tobytes())
    parts.append(np.arange(frames, dtype="<u2").tobytes())

    values = np.empty(spectra * frames * MAX_FREQUENCY_POINTS, dtype=np.complex64)
    for frame in range(frames):
        for frequency in range(MAX_FREQUENCY_POINTS):
            for spectrum in range(spectra):
                index = (frame * MAX_FREQUENCY_POINTS + frequency) * spectra + spectrum
                values[index] = complex(frame + frequency, spectrum)
    interleaved = np.empty(values.size * 2, dtype="<f4")
    interleaved[0::2] = values.real
    interleaved[1::2] = values.imag
    parts.append(interleaved.tobytes())
    path.write_bytes(b"".join(parts))


def test_read_radar_binary_and_segment_features(tmp_path: Path) -> None:
    path = tmp_path / "sample.bin"
    _write_radar_fixture(path)

    recording = read_radar_binary(path)

    assert recording.formatted_time == "2021-01-01"
    assert recording.num_total_frames == 8
    assert len(recording.radargrams) == 8
    assert recording.radargrams[7].shape == (8, 4)
    assert recording.radargrams[7][2, 3] == complex(5, 7)

    features = radar_segment_features(recording, num_frequency_bands=4)
    assert features.shape == (4, 16)
    assert np.isfinite(features).all()


def test_discover_external_radar_corpus_pairs_by_stem(tmp_path: Path) -> None:
    stem = "S001_SES01_CL002_REP003"
    session = tmp_path / "S001" / "SES01"
    for directory in ("audioData", "labels", "radarData"):
        (session / directory).mkdir(parents=True, exist_ok=True)
    (session / "audioData" / f"{stem}.wav").write_bytes(b"audio")
    (session / "labels" / f"{stem}.txt").write_text("kommando", encoding="utf-8")
    _write_radar_fixture(session / "radarData" / f"{stem}.bin")

    manifest = discover_external_radar_corpus(tmp_path)

    assert manifest["sample_id"].tolist() == [stem]
    assert manifest.loc[0, "class_id"] == 1
    assert manifest.loc[0, "repetition"] == 3
    assert manifest.loc[0, "label"] == "kommando"


def test_select_radar_feature_set_preserves_expected_families() -> None:
    features = np.arange(2 * 4 * 8, dtype=np.float32).reshape(2, 4, 8)

    np.testing.assert_array_equal(select_radar_feature_set(features, "all"), features)
    np.testing.assert_array_equal(select_radar_feature_set(features, "s12"), features[:, :, :4])
    np.testing.assert_array_equal(select_radar_feature_set(features, "s32"), features[:, :, 4:])
    np.testing.assert_array_equal(
        select_radar_feature_set(features, "magnitude"),
        features[:, :, [0, 1, 4, 5]],
    )
    np.testing.assert_array_equal(
        select_radar_feature_set(features, "delta"),
        features[:, :, [2, 3, 6, 7]],
    )


def test_external_subject_splits_never_expose_test_subject() -> None:
    users = np.repeat(["S001", "S002"], 6)
    sessions = np.tile(np.repeat(["SES01", "SES02", "SES03"], 2), 2)

    specs = make_external_split_specs(users, sessions, protocol="subject")

    assert len(specs) == 2
    for spec in specs:
        train_users = set(users[spec["train_mask"]])
        val_users = set(users[spec["val_mask"]])
        test_users = set(users[spec["test_mask"]])
        assert train_users == val_users == {spec["train_user"]}
        assert test_users == {spec["test_user"]}
        assert train_users.isdisjoint(test_users)
        assert set(sessions[spec["val_mask"]]) == {"SES03"}


def test_external_session_splits_cover_each_sample_once_as_test() -> None:
    users = np.repeat(["S001", "S002"], 3)
    sessions = np.tile(["SES01", "SES02", "SES03"], 2)

    specs = make_external_split_specs(users, sessions, protocol="session")

    test_counts = np.stack([spec["test_mask"] for spec in specs]).sum(axis=0)
    np.testing.assert_array_equal(test_counts, np.ones(len(users), dtype=np.int64))
