# External Radar-to-HuBERT Replication

## Setup

- Corpus: Wagner et al. stepped-frequency continuous-wave radar command words
- Silent input: S12 and S32 complex radar magnitudes and temporal differences,
  pooled into four relative-time segments
- Teacher: four silence-trimmed HuBERT segments from the paired recording audio
- Evaluation: three recording-session-held-out folds across both corpus subjects
- Classes: 50 German command words; chance accuracy is 2%
- Controls: reversed teacher order, a train-only class-mean HuBERT prototype, and
  a second student trained on within-command residual HuBERT targets

Audio is used only to construct training and evaluation targets. The trained student
receives radar features alone.

## Results

| Test session | Validation session | Best epoch | Radar accuracy | Radar cosine | Class-prototype cosine | Reversed cosine | Order margin | Residual cosine | Residual order margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SES01 | SES02 | 105 | 22.0% | 0.246 | 0.790 | 0.004 | +0.242 | 0.124 | +0.103 |
| SES02 | SES03 | 131 | 25.8% | 0.369 | 0.799 | -0.002 | +0.371 | 0.165 | +0.134 |
| SES03 | SES01 | 125 | 35.6% | 0.354 | 0.795 | -0.026 | +0.379 | 0.165 | +0.132 |

## Aggregate

- Radar command accuracy: **27.8%**
- Radar-to-HuBERT true-order cosine: **0.323**
- Label-only class-prototype cosine: **0.795**
- Reversed-order cosine: **-0.008**
- True-versus-reversed order margin: **+0.331**
- Within-command residual cosine: **0.151**
- Residual true-versus-reversed margin: **+0.123**

## Interpretation boundary

This is an independent cross-session replication of the project method, with a
different lab, language, command inventory, radar system, and speakers from RVTALL.
Because the external corpus contains only two subjects, it does not establish broad
speaker generalization. Positive temporal order margin supports ordered sensor-to-audio
representation alignment. The residual control is the stricter test of
exemplar-specific articulation after command identity is removed. Neither result by
itself establishes phoneme-level recovery.
