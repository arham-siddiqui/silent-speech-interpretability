"""Utilities for the Wagner et al. external radar command-word corpus."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import struct

import numpy as np
import pandas as pd

MAX_FREQUENCY_POINTS = 256
SAMPLE_PATTERN = re.compile(
    r"^(?P<subject>S\d{3})_(?P<session>SES\d{2})_CL(?P<class_id>\d{3})_REP(?P<repetition>\d{3})$"
)


@dataclass(frozen=True)
class RadarRecording:
    formatted_time: str
    displayed_spectrum_indices: np.ndarray
    num_steps: int
    num_total_frames: int
    start_frame_index: int
    stop_frame_index: int
    pll1_frequencies_khz: np.ndarray
    pll2_frequencies_khz: np.ndarray
    pll1_power_levels: np.ndarray
    pll2_power_levels: np.ndarray
    timestamps: np.ndarray
    radargrams: tuple[np.ndarray, ...]


def _read_array(data: bytes, offset: int, count: int, dtype: str) -> tuple[np.ndarray, int]:
    itemsize = np.dtype(dtype).itemsize
    stop = offset + count * itemsize
    if stop > len(data):
        raise ValueError(f"Truncated radar file at byte {offset}: need {count * itemsize} more bytes")
    return np.frombuffer(data, dtype=dtype, count=count, offset=offset).copy(), stop


def read_radar_binary(path: str | Path) -> RadarRecording:
    """Read the corpus binary format and return one complex radargram per spectrum."""
    data = Path(path).read_bytes()
    if len(data) < 21:
        raise ValueError("Radar file is too short to contain a valid header")

    offset = 0
    (time_length,) = struct.unpack_from("<I", data, offset)
    offset += 4
    if time_length > len(data) - offset:
        raise ValueError(f"Invalid formatted-time length: {time_length}")
    formatted_time = data[offset : offset + time_length].decode("utf-8", errors="replace")
    offset += time_length

    num_spectra = data[offset]
    offset += 1
    if not 1 <= num_spectra <= 32:
        raise ValueError(f"Invalid spectrum count: {num_spectra}")
    displayed_spectrum_indices, offset = _read_array(data, offset, num_spectra, "u1")

    if offset + 16 > len(data):
        raise ValueError("Radar file is truncated before frame metadata")
    num_steps, num_total_frames, start_frame_index, stop_frame_index = struct.unpack_from(
        "<IIII", data, offset
    )
    offset += 16
    if not 1 <= num_steps <= MAX_FREQUENCY_POINTS:
        raise ValueError(f"Invalid frequency-step count: {num_steps}")
    if num_total_frames < 1:
        raise ValueError(f"Invalid frame count: {num_total_frames}")

    metadata_arrays = []
    for _ in range(4):
        values, offset = _read_array(data, offset, num_steps, "<i4")
        metadata_arrays.append(values)
    timestamps, offset = _read_array(data, offset, num_total_frames, "<u2")

    complex_count = num_spectra * num_total_frames * MAX_FREQUENCY_POINTS
    parts, offset = _read_array(data, offset, complex_count * 2, "<f4")
    if offset != len(data):
        raise ValueError(f"Unexpected {len(data) - offset} trailing bytes in radar file")
    complex_values = parts[0::2] + 1j * parts[1::2]

    radargrams = []
    for spectrum_index in range(num_spectra):
        spectrum = complex_values[spectrum_index::num_spectra]
        spectrum = spectrum.reshape(num_total_frames, MAX_FREQUENCY_POINTS)
        radargrams.append(spectrum[:, :num_steps].astype(np.complex64, copy=False))

    return RadarRecording(
        formatted_time=formatted_time,
        displayed_spectrum_indices=displayed_spectrum_indices,
        num_steps=num_steps,
        num_total_frames=num_total_frames,
        start_frame_index=start_frame_index,
        stop_frame_index=stop_frame_index,
        pll1_frequencies_khz=metadata_arrays[0],
        pll2_frequencies_khz=metadata_arrays[1],
        pll1_power_levels=metadata_arrays[2],
        pll2_power_levels=metadata_arrays[3],
        timestamps=timestamps,
        radargrams=tuple(radargrams),
    )


def discover_external_radar_corpus(root: str | Path) -> pd.DataFrame:
    """Build a deterministic manifest and reject missing or unpaired sample files."""
    root = Path(root)
    radar_files = sorted(root.glob("S*/SES*/radarData/*.bin"))
    if not radar_files:
        raise FileNotFoundError(f"No radar files found under {root}")

    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    for radar_path in radar_files:
        stem = radar_path.stem
        match = SAMPLE_PATTERN.fullmatch(stem)
        if match is None:
            raise ValueError(f"Unexpected radar filename: {radar_path.name}")
        if stem in seen:
            raise ValueError(f"Duplicate sample identifier: {stem}")
        seen.add(stem)

        subject = match.group("subject")
        session = match.group("session")
        sample_root = root / subject / session
        audio_path = sample_root / "audioData" / f"{stem}.wav"
        label_path = sample_root / "labels" / f"{stem}.txt"
        missing = [path for path in (audio_path, label_path) if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Unpaired sample {stem}; missing: {', '.join(map(str, missing))}")
        label = label_path.read_text(encoding="utf-8").strip()
        if not label:
            raise ValueError(f"Empty label file: {label_path}")

        rows.append(
            {
                "sample_id": stem,
                "user_id": subject,
                "session_id": session,
                "class_id": int(match.group("class_id")) - 1,
                "repetition": int(match.group("repetition")),
                "group_name": stem,
                "label": label,
                "audio_path": str(audio_path.resolve()),
                "radar_path": str(radar_path.resolve()),
                "label_path": str(label_path.resolve()),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(
        ["user_id", "session_id", "class_id", "repetition"], ignore_index=True
    )
    if (manifest["class_id"] < 0).any():
        raise ValueError("Class identifiers must start at CL001")
    return manifest


def _frequency_band_means(values: np.ndarray, num_bands: int) -> np.ndarray:
    bands = np.array_split(values, num_bands, axis=1)
    if any(band.shape[1] == 0 for band in bands):
        raise ValueError(f"num_bands={num_bands} exceeds available frequency steps={values.shape[1]}")
    return np.stack([band.mean(axis=1) for band in bands], axis=1)


def radar_segment_features(
    recording: RadarRecording,
    *,
    spectrum_indices: tuple[int, ...] = (1, 7),
    num_segments: int = 4,
    num_frequency_bands: int = 16,
) -> np.ndarray:
    """Create relative-time features from the S12/S32 radargrams used by the corpus paper."""
    if recording.num_total_frames < num_segments:
        raise ValueError(
            f"Need at least {num_segments} radar frames, found {recording.num_total_frames}"
        )
    if max(spectrum_indices) >= len(recording.radargrams):
        raise ValueError(
            f"Requested spectrum {max(spectrum_indices)}, file has {len(recording.radargrams)} spectra"
        )

    per_frame = []
    for spectrum_index in spectrum_indices:
        magnitude = np.log1p(np.abs(recording.radargrams[spectrum_index])).astype(np.float32)
        scale = magnitude.std(axis=0, keepdims=True)
        standardized = (magnitude - magnitude.mean(axis=0, keepdims=True)) / np.maximum(scale, 1e-6)
        delta = np.diff(standardized, axis=0, prepend=standardized[:1])
        per_frame.extend(
            [
                _frequency_band_means(standardized, num_frequency_bands),
                _frequency_band_means(np.abs(delta), num_frequency_bands),
            ]
        )
    frame_features = np.concatenate(per_frame, axis=1)
    segments = np.array_split(frame_features, num_segments, axis=0)
    return np.stack([segment.mean(axis=0) for segment in segments]).astype(np.float32)


def select_radar_feature_set(features: np.ndarray, feature_set: str) -> np.ndarray:
    """Select from [S12 magnitude, S12 delta, S32 magnitude, S32 delta] columns."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 3 or features.shape[2] % 4 != 0:
        raise ValueError("Radar features must have shape [samples, segments, 4 * bands]")
    width = features.shape[2] // 4
    columns = {
        "all": np.arange(4 * width),
        "s12": np.arange(2 * width),
        "s32": np.arange(2 * width, 4 * width),
        "magnitude": np.concatenate([np.arange(width), np.arange(2 * width, 3 * width)]),
        "delta": np.concatenate([np.arange(width, 2 * width), np.arange(3 * width, 4 * width)]),
    }
    if feature_set not in columns:
        raise ValueError(f"Unknown feature set {feature_set!r}; choose from {sorted(columns)}")
    return features[:, :, columns[feature_set]].astype(np.float32)


def make_external_split_specs(
    user_ids: np.ndarray,
    session_ids: np.ndarray,
    *,
    protocol: str,
    subject_val_session: str = "SES03",
) -> list[dict[str, object]]:
    """Create disjoint session- or subject-held-out external evaluation masks."""
    users = np.asarray(user_ids).astype(str)
    sessions = np.asarray(session_ids).astype(str)
    if users.shape != sessions.shape or users.ndim != 1:
        raise ValueError("user_ids and session_ids must be aligned one-dimensional arrays")
    unique_sessions = sorted(np.unique(sessions))
    unique_users = sorted(np.unique(users))
    if len(unique_sessions) != 3:
        raise ValueError(f"Expected three sessions, found {unique_sessions}")
    if subject_val_session not in unique_sessions:
        raise ValueError(f"Unknown subject validation session: {subject_val_session}")

    specs: list[dict[str, object]] = []
    if protocol == "session":
        for fold, test_session in enumerate(unique_sessions):
            val_session = unique_sessions[(fold + 1) % len(unique_sessions)]
            train_session = next(
                session
                for session in unique_sessions
                if session not in {test_session, val_session}
            )
            specs.append(
                {
                    "fold": fold,
                    "train_mask": sessions == train_session,
                    "val_mask": sessions == val_session,
                    "test_mask": sessions == test_session,
                    "train_session": train_session,
                    "val_session": val_session,
                    "test_session": test_session,
                    "train_user": "ALL",
                    "test_user": "ALL",
                }
            )
    elif protocol == "subject":
        if len(unique_users) != 2:
            raise ValueError(f"Expected two subjects for subject transfer, found {unique_users}")
        for fold, test_user in enumerate(unique_users):
            train_user = next(user for user in unique_users if user != test_user)
            source_mask = users == train_user
            specs.append(
                {
                    "fold": fold,
                    "train_mask": source_mask & (sessions != subject_val_session),
                    "val_mask": source_mask & (sessions == subject_val_session),
                    "test_mask": users == test_user,
                    "train_session": "+".join(
                        session for session in unique_sessions if session != subject_val_session
                    ),
                    "val_session": subject_val_session,
                    "test_session": "ALL",
                    "train_user": train_user,
                    "test_user": test_user,
                }
            )
    else:
        raise ValueError("protocol must be 'session' or 'subject'")

    for spec in specs:
        train_mask = spec["train_mask"]
        val_mask = spec["val_mask"]
        test_mask = spec["test_mask"]
        if not train_mask.any() or not val_mask.any() or not test_mask.any():
            raise ValueError(f"Empty split in fold {spec['fold']}")
        if (train_mask & val_mask).any() or (train_mask & test_mask).any() or (val_mask & test_mask).any():
            raise ValueError(f"Overlapping split in fold {spec['fold']}")
    return specs
