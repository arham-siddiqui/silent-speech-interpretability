# RVTALL Audio Manifest Audit

The local RVTALL audio discovery and alignment step completed successfully.

## Coverage

| Metric | Value |
|---|---:|
| Manifest rows | 599 |
| WAV files discovered | 5300 |
| Unique speaker/group audio pairs | 596 |
| Rows with an audio path | 596 |
| Coverage | 99.5% |
| Exact Kinect repetition matches | 596 |
| Latest-repetition fallbacks | 0 |
| Missing pairs | 3 |

Missing pairs: `5::sentences5`, `5::word7`, `18::sentences8`

## Alignment Rule

Each manifest pair is matched to the WAV repetition synchronized with the lip
embedding selected by the repository's current duplicate-pair index. Concretely,
`video_N` maps to `audio_proc_N.wav`. A latest-available repetition is used only
when a pair has audio but no reference repetition.

The generated manifest is `artifacts/manifest_with_audio.csv`. It contains machine-local absolute
paths and is intentionally excluded from Git; this audit report is pushable.
