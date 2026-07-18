# HuBERT Teacher-Student CV Results

This experiment trains the silent-sensor student against mean-pooled final-layer
HuBERT targets using fold-specific, encoder-disjoint sensor embeddings.

## Setup

- Teacher: `facebook/hubert-base-ls960`
- Teacher targets: `artifacts/teacher_targets/facebook_hubert-base-ls960_targets.npz`
- Sensor modalities: lip, UWB, mmWave, and laser
- Mouth: excluded, matching the strict fusion baseline
- Evaluation: 5-fold speaker-disjoint CV
- Maximum epochs per fold: 100, with validation-loss early stopping
- HuBERT targets are centered using training-fold statistics before normalization
- Audio is used only to create fixed teacher targets and is absent from student inference

## Per-Fold Results

| Fold | Train | Validation | Test | Validation Accuracy | Student Accuracy | Fusion Baseline | Delta (pp) | Target Cosine | Target MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 392 | 30 | 117 | 86.7% | 67.5% | 62.4% | +5.1 | 0.443 | 1.1138 |
| 1 | 391 | 58 | 90 | 77.6% | 61.1% | 57.8% | +3.3 | 0.405 | 1.1895 |
| 2 | 379 | 58 | 102 | 81.0% | 62.7% | 61.8% | +1.0 | 0.450 | 1.0993 |
| 3 | 366 | 58 | 115 | 79.3% | 74.8% | 75.7% | -0.9 | 0.439 | 1.1218 |
| 4 | 366 | 58 | 115 | 70.7% | 53.9% | 61.7% | -7.8 | 0.413 | 1.1733 |

## Aggregate

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Student test accuracy | 64.0% | 7.8% |
| Strict fusion test accuracy | 63.9% | 6.8% |
| Paired accuracy delta | +0.1 pp | 5.0 pp |
| Residual-HuBERT cosine | 0.430 | 0.020 |
| Test target MSE | 1.1395 | 0.0395 |

The student classifier is **0.1 percentage points above** the
current validation-weighted strict fusion baseline and wins on 3 of 5 folds.
The paired fold deltas vary substantially, so this small mean difference is not evidence
of a reliable improvement by itself. Target MSE separately measures how well silent
sensors recover the teacher representation.

The train-mean residual direction scores -0.001 cosine on held-out
speakers, compared with 0.430 for the student. Centering removes HuBERT's
dominant shared direction, so this measures recovery of utterance-varying structure.

## Interpretation Boundary

These targets are utterance-level mean-pooled HuBERT states. They establish the first
real audio-teacher baseline, but they do not provide frame-level articulatory or
syllable interpretation. A relative-segment temporal HuBERT comparison is now complete;
frame-level silent-sensor activations and SPARC or Sylber remain follow-up experiments.
