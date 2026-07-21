import pytest

from silent_speech_interpretability.data.prompts import (
    arpabet_words,
    ctc_text,
    expected_prompt_rows,
    prompt_for,
    sentence_variants,
    strip_phone_stress,
)


def test_prompt_map_has_all_twenty_speakers_and_thirty_groups():
    rows = expected_prompt_rows()
    assert len(rows) == 600
    assert len({(row["user_id"], row["group_name"]) for row in rows}) == 600


def test_sentence_cohorts_are_speaker_specific():
    assert prompt_for(1, "sentences7")["transcript"] == "The staff sanitized the sickroom."
    assert prompt_for(3, "sentences7")["transcript"] == "I am having trouble breathing."
    assert prompt_for(20, "sentences7")["transcript"] == "Need emergency treatment at shock stage."
    assert prompt_for(8, "sentences8")["transcript"] == "Is there a doctor here?"
    assert prompt_for(1, "sentences7", cohort_override="breathing")["transcript"] == "I am having trouble breathing."
    assert sentence_variants(10)["emergency"] == "Don't worry about falling."


def test_vowel_targets_and_group_normalization():
    assert prompt_for("2", "vowel_1")["vowel_arpabet"] == "AE"
    assert prompt_for(2, "word_4")["transcript"] == "ambulance"


def test_unknown_prompt_raises():
    with pytest.raises(KeyError):
        prompt_for(1, "word16")


def test_ctc_and_pronunciation_normalization():
    assert ctc_text("I think I'm having a heart attack.") == "I THINK I'M HAVING A HEART ATTACK"
    sickroom = dict(arpabet_words("the sickroom"))["sickroom"]
    assert [strip_phone_stress(phone) for phone in sickroom] == ["S", "IH", "K", "R", "UW", "M"]
