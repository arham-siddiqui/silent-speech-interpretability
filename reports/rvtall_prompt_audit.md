# RVTALL Prompt Audit

The anonymous RVTALL corpus labels are mapped to the prompts published in Table 5 of the
dataset paper: https://pmc.ncbi.nlm.nih.gov/articles/PMC10719268/#Tab5

## Coverage

- Expected speaker/group prompts: **600**.
- Audio-manifest rows enriched: **599**.
- Missing expected pairs: **1** (user 4 / vowel5).
- Unexpected pairs: **0** (None).
- Rows with an existing audio file: **596**.
- Local audio cohort overrides loaded: **20**.

## Speaker-Specific Sentence Classes

The publication assigns different text to sentence indices 7, 9, and 10 for three
participant cohorts. These must not be treated as one transcript per class.

The source table's participant IDs are preserved in the canonical mapping. When
`metadata/rvtall_audio_prompt_cohorts.csv` exists, the working manifest uses its audited
audio-folder mapping because local processed folder IDs do not match the published IDs.

| Group | Cohort | Transcript | Speakers |
|---|---|---|---:|
| sentences10 | breathing | My heart is failing. | 7 |
| sentences10 | emergency | Don't worry about falling. | 8 |
| sentences10 | sanitation | Don't worry about bleeding. | 5 |
| sentences7 | breathing | I am having trouble breathing. | 7 |
| sentences7 | emergency | Need emergency treatment at shock stage. | 8 |
| sentences7 | sanitation | The staff sanitized the sickroom. | 5 |
| sentences9 | breathing | I think I'm having a heart attack. | 7 |
| sentences9 | emergency | He need a rescue for a heart attack. | 8 |
| sentences9 | sanitation | Medical care is important. | 5 |

The paper prints `failling`; the normalized alignment transcript uses `failing` while
preserving the intended spoken phrase. Vowels retain their published IPA category and an
explicit ARPAbet target for later alignment.
