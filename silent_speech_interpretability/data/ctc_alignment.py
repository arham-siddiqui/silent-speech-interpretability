"""Small, dependency-light CTC forced-alignment utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TokenSpan:
    target_index: int
    token_id: int
    start_frame: int
    end_frame: int
    mean_log_probability: float


def ctc_viterbi_align(log_probs: np.ndarray, target_ids: list[int], blank_id: int) -> list[TokenSpan]:
    """Align a known token sequence to frame log-probabilities with CTC Viterbi DP."""
    scores = np.asarray(log_probs, dtype=np.float64)
    if scores.ndim != 2 or not target_ids:
        raise ValueError("log_probs must be [frames, vocabulary] and target_ids must be nonempty")
    if min(target_ids) < 0 or max(target_ids) >= scores.shape[1]:
        raise ValueError("target token is outside the vocabulary")

    states = np.full(2 * len(target_ids) + 1, blank_id, dtype=np.int64)
    states[1::2] = target_ids
    previous = np.full(len(states), -np.inf)
    previous[0] = scores[0, blank_id]
    previous[1] = scores[0, target_ids[0]]
    backpointers = np.full((len(scores), len(states)), -1, dtype=np.int8)

    for frame in range(1, len(scores)):
        current = np.full(len(states), -np.inf)
        for state, token in enumerate(states):
            candidates = [(previous[state], 0)]
            if state >= 1:
                candidates.append((previous[state - 1], 1))
            if state >= 2 and state % 2 == 1 and states[state] != states[state - 2]:
                candidates.append((previous[state - 2], 2))
            best_score, step = max(candidates, key=lambda item: item[0])
            current[state] = best_score + scores[frame, token]
            backpointers[frame, state] = step
        previous = current

    final_candidates = [(previous[-1], len(states) - 1), (previous[-2], len(states) - 2)]
    score, state = max(final_candidates)
    if not np.isfinite(score):
        raise ValueError("No valid CTC alignment path")
    state_path = np.empty(len(scores), dtype=np.int64)
    state_path[-1] = state
    for frame in range(len(scores) - 1, 0, -1):
        step = int(backpointers[frame, state])
        if step < 0:
            raise ValueError("Incomplete CTC backtrace")
        state -= step
        state_path[frame - 1] = state

    spans = []
    for target_index, token_id in enumerate(target_ids):
        target_state = 2 * target_index + 1
        frames = np.flatnonzero(state_path == target_state)
        if not len(frames):
            raise ValueError(f"Target token {target_index} has no aligned frame")
        spans.append(
            TokenSpan(
                target_index=target_index,
                token_id=int(token_id),
                start_frame=int(frames[0]),
                end_frame=int(frames[-1] + 1),
                mean_log_probability=float(scores[frames, token_id].mean()),
            )
        )
    return spans


def levenshtein_distance(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for left_index, left_value in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_value in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_value != right_value),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    return levenshtein_distance(reference, hypothesis) / max(len(reference), 1)
