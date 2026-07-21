"""Authoritative speaker-aware RVTALL prompt mapping."""

from __future__ import annotations

import re
from functools import lru_cache


SOURCE_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC10719268/#Tab5"

VOWELS = {
    1: ("[ae]", "AE"),
    2: ("[i]", "IY"),
    3: ("[schwa]", "AH"),
    4: ("[open-o]", "AO"),
    5: ("[u]", "UW"),
}

WORDS = {
    1: "order",
    2: "assist",
    3: "help",
    4: "ambulance",
    5: "bleed",
    6: "fall",
    7: "shock",
    8: "medical",
    9: "sanitize",
    10: "doctor",
    11: "accident",
    12: "rescue",
    13: "emergency",
    14: "heart",
    15: "break",
}

COMMON_SENTENCES = {
    1: "I need help.",
    2: "Call for an ambulance.",
    3: "The building's on fire.",
    4: "Can you smell smoke?",
    5: "Where's the fire escape?",
    6: "There's been an accident.",
    8: "Is there a doctor here?",
}

SENTENCE_COHORTS = (
    (
        "sanitation",
        frozenset({1, 2, 4, 6, 7}),
        {
            7: "The staff sanitized the sickroom.",
            9: "Medical care is important.",
            10: "Don't worry about bleeding.",
        },
    ),
    (
        "breathing",
        frozenset({3, 8, 9, 10, 11, 12, 13}),
        {
            7: "I am having trouble breathing.",
            9: "I think I'm having a heart attack.",
            10: "My heart is failing.",
        },
    ),
    (
        "emergency",
        frozenset({5, 14, 15, 16, 17, 18, 19, 20}),
        {
            7: "Need emergency treatment at shock stage.",
            9: "He need a rescue for a heart attack.",
            10: "Don't worry about falling.",
        },
    ),
)


def normalize_group_name(group_name: str) -> str:
    return str(group_name).lower().replace("_", "")


def prompt_for(
    user_id: int | str,
    group_name: str,
    *,
    cohort_override: str | None = None,
) -> dict[str, str | int]:
    user = int(user_id)
    group = normalize_group_name(group_name)
    match = re.fullmatch(r"(vowel|word|sentences)(\d+)", group)
    if match is None:
        raise KeyError(f"Unknown RVTALL group: {group_name}")
    kind, raw_index = match.groups()
    index = int(raw_index)
    if kind == "vowel":
        display, arpabet = VOWELS[index]
        return {
            "prompt_type": "vowel",
            "prompt_index": index,
            "transcript": display,
            "spoken_text": "",
            "vowel_arpabet": arpabet,
            "prompt_cohort": "all",
        }
    if kind == "word":
        transcript = WORDS[index]
        return {
            "prompt_type": "word",
            "prompt_index": index,
            "transcript": transcript,
            "spoken_text": transcript,
            "vowel_arpabet": "",
            "prompt_cohort": "all",
        }
    if index in COMMON_SENTENCES:
        transcript = COMMON_SENTENCES[index]
        cohort = "all"
    else:
        if cohort_override is None:
            cohort, _users, prompts = next(
                (name, users, prompts)
                for name, users, prompts in SENTENCE_COHORTS
                if user in users
            )
        else:
            cohort, _users, prompts = next(
                (name, users, prompts)
                for name, users, prompts in SENTENCE_COHORTS
                if name == cohort_override
            )
        transcript = prompts[index]
    return {
        "prompt_type": "sentence",
        "prompt_index": index,
        "transcript": transcript,
        "spoken_text": transcript,
        "vowel_arpabet": "",
        "prompt_cohort": cohort,
    }


def expected_prompt_rows() -> list[dict[str, str | int]]:
    rows = []
    for user_id in range(1, 21):
        for prefix, count in (("vowel", 5), ("word", 15), ("sentences", 10)):
            for index in range(1, count + 1):
                group_name = f"{prefix}{index}"
                rows.append({"user_id": user_id, "group_name": group_name, **prompt_for(user_id, group_name)})
    return rows


def sentence_variants(index: int) -> dict[str, str]:
    if index not in {7, 9, 10}:
        raise KeyError(f"Sentence {index} does not vary by cohort")
    return {name: prompts[index] for name, _users, prompts in SENTENCE_COHORTS}


def ctc_text(text: str) -> str:
    """Normalize published prompt text to the Wav2Vec2 English CTC alphabet."""
    words = re.findall(r"[a-z']+", str(text).lower())
    return " ".join(words).upper()


@lru_cache(maxsize=1)
def _cmu_dictionary() -> dict[str, list[list[str]]]:
    try:
        import cmudict
    except ImportError as exc:
        raise RuntimeError("Install the alignment extra to generate pronunciations") from exc
    return cmudict.dict()


def arpabet_words(text: str) -> list[tuple[str, list[str]]]:
    """Return deterministic CMU pronunciations for each normalized prompt token."""
    dictionary = _cmu_dictionary()
    overrides = {
        "sickroom": dictionary["sick"][0] + dictionary["room"][0],
    }
    output = []
    for word in re.findall(r"[a-z']+", str(text).lower()):
        pronunciations = dictionary.get(word)
        if pronunciations:
            phones = pronunciations[0]
        elif word in overrides:
            phones = overrides[word]
        else:
            raise KeyError(f"No CMU pronunciation for {word!r}")
        output.append((word, list(phones)))
    return output


def strip_phone_stress(phone: str) -> str:
    return re.sub(r"\d+$", "", phone)
