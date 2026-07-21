# RVTALL Pronunciation Audit

- Manifest rows: **599**.
- Unique lexical tokens: **64**.
- Stress-stripped ARPAbet inventory: **34** phones.
- Non-vowel rows with nonempty CTC text: **500**.
- Rows with a nonempty pronunciation: **599**.
- Minimum/median/maximum phones per prompt: **1 / 7 / 28**.

The lexicon uses the first CMUdict pronunciation for reproducibility. `sickroom` is the
only token absent from CMUdict and is composed from the entries for `sick` and `room`.
Lexical stress is retained for alignment metadata and removed for broad phonetic probes.
