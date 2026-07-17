# True CV Error Analysis

This report focuses on `validation_weighted` fusion, the current strict-CV baseline.

![Per-class accuracy](figures/true_cv_per_class_accuracy.svg)

## Hardest Classes

| Class ID | Accuracy | Correct | Samples |
|---:|---:|---:|---:|
| 12 | 27.8% | 5 | 18 |
| 10 | 37.5% | 6 | 16 |
| 22 | 38.9% | 7 | 18 |
| 18 | 42.1% | 8 | 19 |
| 11 | 43.8% | 7 | 16 |
| 20 | 44.4% | 8 | 18 |
| 25 | 47.1% | 8 | 17 |
| 16 | 47.4% | 9 | 19 |
| 21 | 50.0% | 9 | 18 |
| 13 | 55.6% | 10 | 18 |

## Strongest Classes

| Class ID | Accuracy | Correct | Samples |
|---:|---:|---:|---:|
| 0 | 100.0% | 19 | 19 |
| 2 | 100.0% | 19 | 19 |
| 8 | 100.0% | 18 | 18 |
| 5 | 88.9% | 16 | 18 |
| 24 | 88.9% | 16 | 18 |
| 6 | 84.2% | 16 | 19 |
| 4 | 83.3% | 15 | 18 |
| 19 | 82.4% | 14 | 17 |
| 9 | 78.9% | 15 | 19 |
| 17 | 68.4% | 13 | 19 |

## How To Use This

The hardest classes are the first targets for confusion-matrix review and modality
ablation. If a class is weak under validation-weighted fusion but strong under one
individual modality, that class is a candidate for class-conditional or reliability-aware
fusion improvements.
