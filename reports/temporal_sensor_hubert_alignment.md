# Temporal Silent-Sensor To HuBERT Alignment

Fold-specific lip, laser, mmWave, and UWB encoders now expose four relative-time
activation segments. A shared segment student maps their concatenated temporal
representations to the four-segment HuBERT teacher.

## Setup

- Modalities: lip, laser, mmwave, uwb
- Silent input: four 128-D segments per modality, averaged across repetitions
- Teacher: four silence-trimmed 768-D HuBERT segments
- Evaluation: five speaker-disjoint, encoder-disjoint folds
- Controls: reversed and shifted teacher segment order

## Results

| Fold | Temporal-Sensor Accuracy | Fixed-Embedding Accuracy | Sensor Segment Cosine | Fixed Segment Cosine | Reversed Cosine | Order Margin |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 57.3% | 65.8% | 0.398 | 0.361 | 0.081 | +0.317 |
| 1 | 45.6% | 57.8% | 0.374 | 0.293 | 0.015 | +0.359 |
| 2 | 50.0% | 65.7% | 0.395 | 0.369 | 0.057 | +0.338 |
| 3 | 59.1% | 78.3% | 0.385 | 0.347 | 0.010 | +0.375 |
| 4 | 37.4% | 54.8% | 0.351 | 0.357 | 0.075 | +0.277 |

## Aggregate

- Temporal-sensor class accuracy: **49.9% +/- 8.9%**.
- Fixed-embedding temporal-student accuracy: **64.5%**.
- Temporal-sensor true-order cosine: **0.381**.
- Fixed-embedding true-order cosine: **0.346**.
- Temporal-sensor reversed-order cosine: **0.047**.
- Temporal-sensor true-versus-reversed margin: **+0.333**.

This tests whether temporal silent-sensor states contain ordered HuBERT information. It
does not imply frame-exact synchronization because each modality is pooled into relative
regions and repetitions are averaged within speaker/utterance pairs.
