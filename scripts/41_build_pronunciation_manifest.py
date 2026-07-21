#!/usr/bin/env python3
"""Add canonical CTC text and ARPAbet pronunciations to RVTALL prompts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from silent_speech_interpretability.data.prompts import arpabet_words, ctc_text, strip_phone_stress


def _pronunciation(row) -> tuple[list[dict[str, object]], list[str]]:
    if row.prompt_type == "vowel":
        phones = [str(row.vowel_arpabet)]
        return [{"word": str(row.transcript), "phones": phones}], phones
    words = [{"word": word, "phones": phones} for word, phones in arpabet_words(row.spoken_text)]
    phones = [phone for item in words for phone in item["phones"]]
    return words, phones


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="artifacts/manifest_with_transcripts.csv")
    parser.add_argument("--output", default="artifacts/manifest_with_pronunciations.csv")
    parser.add_argument("--lexicon-output", default="metadata/rvtall_pronunciation_lexicon.csv")
    parser.add_argument("--report-output", default="reports/rvtall_pronunciation_audit.md")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest).fillna("")
    records = []
    lexicon: dict[str, list[str]] = {}
    inventory = set()
    for row in manifest.itertuples(index=False):
        words, stressed = _pronunciation(row)
        unstressed = [strip_phone_stress(phone) for phone in stressed]
        inventory.update(unstressed)
        for item in words:
            lexicon.setdefault(str(item["word"]), list(item["phones"]))
        records.append(
            {
                "ctc_text": ctc_text(row.spoken_text),
                "word_pronunciations": json.dumps(words, separators=(",", ":")),
                "arpabet": " ".join(stressed),
                "arpabet_no_stress": " ".join(unstressed),
                "phoneme_count": len(stressed),
            }
        )
    enriched = pd.concat([manifest.reset_index(drop=True), pd.DataFrame(records)], axis=1)
    lexicon_rows = [
        {
            "token": word,
            "arpabet": " ".join(phones),
            "arpabet_no_stress": " ".join(strip_phone_stress(phone) for phone in phones),
        }
        for word, phones in sorted(lexicon.items())
    ]
    output = Path(args.output)
    lexicon_output = Path(args.lexicon_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    lexicon_output.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_csv(output, index=False)
    pd.DataFrame(lexicon_rows).to_csv(lexicon_output, index=False)
    report = f"""# RVTALL Pronunciation Audit

- Manifest rows: **{len(enriched)}**.
- Unique lexical tokens: **{len(lexicon_rows)}**.
- Stress-stripped ARPAbet inventory: **{len(inventory)}** phones.
- Non-vowel rows with nonempty CTC text: **{((enriched.prompt_type != 'vowel') & enriched.ctc_text.ne('')).sum()}**.
- Rows with a nonempty pronunciation: **{enriched.arpabet.ne('').sum()}**.
- Minimum/median/maximum phones per prompt: **{enriched.phoneme_count.min()} / {enriched.phoneme_count.median():.0f} / {enriched.phoneme_count.max()}**.

The lexicon uses the first CMUdict pronunciation for reproducibility. `sickroom` is the
only token absent from CMUdict and is composed from the entries for `sick` and `room`.
Lexical stress is retained for alignment metadata and removed for broad phonetic probes.
"""
    Path(args.report_output).write_text(report, encoding="utf-8")
    print(f"Saved pronunciation manifest to {output}")
    print(f"Saved pronunciation audit to {args.report_output}")


if __name__ == "__main__":
    main()
