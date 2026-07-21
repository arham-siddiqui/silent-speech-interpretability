"""Phonetic feature definitions and interval pooling for alignment probes."""

from __future__ import annotations

import re

import numpy as np


MANNER_FEATURES = ("vowel", "stop", "fricative", "affricate", "nasal", "liquid", "glide", "silence")

PHONE_MANNER = {
    **{phone: "vowel" for phone in "AA AE AH AO AW AY EH ER EY IH IY OW OY UH UW".split()},
    **{phone: "stop" for phone in "P B T D K G".split()},
    **{phone: "fricative" for phone in "F V TH DH S Z SH ZH HH".split()},
    **{phone: "affricate" for phone in "CH JH".split()},
    **{phone: "nasal" for phone in "M N NG".split()},
    **{phone: "liquid" for phone in "L R".split()},
    **{phone: "glide" for phone in "W Y".split()},
}


def phone_manner(phone: str) -> str:
    normalized = re.sub(r"\d+$", "", phone)
    if normalized not in PHONE_MANNER:
        raise KeyError(f"Unknown ARPAbet phone: {phone}")
    return PHONE_MANNER[normalized]


def interval_occupancy(
    intervals: list[tuple[float, float, str]],
    start_seconds: float,
    end_seconds: float,
    num_segments: int,
    feature_names: tuple[str, ...] = MANNER_FEATURES,
) -> np.ndarray:
    """Pool labeled time intervals into relative-segment occupancy fractions."""
    if end_seconds <= start_seconds or num_segments < 1:
        raise ValueError("A positive interval and at least one segment are required")
    boundaries = np.linspace(start_seconds, end_seconds, num_segments + 1)
    values = np.zeros((num_segments, len(feature_names)), dtype=np.float32)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    non_silence = np.zeros(num_segments, dtype=np.float32)
    for interval_start, interval_end, feature in intervals:
        if feature not in feature_index:
            continue
        for segment in range(num_segments):
            overlap = max(0.0, min(interval_end, boundaries[segment + 1]) - max(interval_start, boundaries[segment]))
            fraction = overlap / (boundaries[segment + 1] - boundaries[segment])
            values[segment, feature_index[feature]] += fraction
            if feature != "silence":
                non_silence[segment] += fraction
    if "silence" in feature_index:
        values[:, feature_index["silence"]] = np.maximum(0.0, 1.0 - np.minimum(non_silence, 1.0))
    return np.clip(values, 0.0, 1.0)
