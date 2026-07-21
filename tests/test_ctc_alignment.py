import numpy as np

from silent_speech_interpretability.data.ctc_alignment import character_error_rate, ctc_viterbi_align


def test_ctc_viterbi_alignment_handles_repeated_tokens():
    probabilities = np.full((7, 3), 0.01)
    path = [0, 1, 1, 0, 1, 2, 0]
    for frame, token in enumerate(path):
        probabilities[frame, token] = 0.98
    spans = ctc_viterbi_align(np.log(probabilities), [1, 1, 2], blank_id=0)
    assert [(span.start_frame, span.end_frame) for span in spans] == [(1, 3), (4, 5), (5, 6)]


def test_character_error_rate():
    assert character_error_rate("HELP", "HEP") == 0.25
    assert character_error_rate("SAME", "SAME") == 0.0
