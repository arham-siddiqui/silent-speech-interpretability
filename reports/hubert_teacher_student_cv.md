# HuBERT Teacher-Student CV Results

This experiment trains the silent-sensor student against mean-pooled final-layer
HuBERT targets using fold-specific, encoder-disjoint sensor embeddings.

## Setup

- Teacher: `facebook/hubert-base-ls960`
- Teacher targets: `artifacts/teacher_targets/facebook_hubert-base-ls960_targets.npz`
- Sensor modalities: lip, UWB, mmWave, and laser
- Mouth: excluded, matching the strict fusion baseline
- Evaluation: 5-fold speaker-disjoint CV
- Maximum epochs per fold: 30, with validation-loss early stopping
- Audio is used only to create fixed teacher targets and is absent from student inference

## Per-Fold Results

| Fold | Train | Validation | Test | Validation Accuracy | Student Accuracy | Fusion Baseline | Delta (pp) | Test Target MSE |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 392 | 30 | 117 | 86.7% | 68.4% | 62.4% | +6.0 | 0.9732 |
| 1 | 391 | 58 | 90 | 74.1% | 58.9% | 57.8% | +1.1 | 0.9191 |
| 2 | 379 | 58 | 102 | 77.6% | 60.8% | 61.8% | -1.0 | 1.0582 |
| 3 | 366 | 58 | 115 | 81.0% | 80.0% | 75.7% | +4.3 | 1.0607 |
| 4 | 366 | 58 | 115 | 75.9% | 54.8% | 61.7% | -7.0 | 0.9884 |

## Aggregate

| Metric | Mean | Standard Deviation |
|---|---:|---:|
| Student test accuracy | 64.6% | 9.9% |
| Strict fusion test accuracy | 63.9% | 6.8% |
| Paired accuracy delta | +0.7 pp | 5.1 pp |
| Test target MSE | 0.9999 | 0.0602 |

The student classifier is **0.7 percentage points above** the
current validation-weighted strict fusion baseline and wins on 3 of 5 folds.
The paired fold deltas vary substantially, so this small mean difference is not evidence
of a reliable improvement by itself. Target MSE separately measures how well silent
sensors recover the teacher representation.

## Interpretation Boundary

These targets are utterance-level mean-pooled HuBERT states. They establish the first
real audio-teacher baseline, but they do not yet provide frame-level articulatory or
syllable interpretation. Temporal HuBERT, SPARC, or Sylber targets remain follow-up
experiments.
