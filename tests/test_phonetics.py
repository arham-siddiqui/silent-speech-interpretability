import numpy as np

from silent_speech_interpretability.data.phonetics import PHONE_MANNER, MANNER_FEATURES, arpabet_to_ipa, interval_occupancy, phone_manner


def test_phone_manner_strips_stress():
    assert phone_manner("AE1") == "vowel"
    assert phone_manner("NG") == "nasal"
    assert arpabet_to_ipa("AH0") == "ə"
    assert arpabet_to_ipa("AH1") == "ʌ"
    assert arpabet_to_ipa("JH") == "dʒ"
    assert all(arpabet_to_ipa(phone) for phone in PHONE_MANNER)


def test_interval_occupancy_tracks_silence_and_overlap():
    values = interval_occupancy([(0.5, 1.5, "vowel")], 0.0, 2.0, 2)
    vowel = MANNER_FEATURES.index("vowel")
    silence = MANNER_FEATURES.index("silence")
    assert np.allclose(values[:, vowel], [0.5, 0.5])
    assert np.allclose(values[:, silence], [0.5, 0.5])
