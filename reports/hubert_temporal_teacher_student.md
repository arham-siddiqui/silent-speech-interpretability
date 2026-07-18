# Temporal HuBERT Teacher-Student Results

The audio teacher was silence-trimmed and mean-pooled into 4 ordered relative-time
segments. The silent student predicts the resulting `4 x 768` HuBERT signature
from lip, UWB, mmWave, and laser embeddings; audio remains absent at inference.

## Design

- Evaluation uses the existing five speaker-disjoint, encoder-disjoint folds.
- Teacher centering is fitted separately on each training fold.
- True segment order is compared with reversed and one-step-shifted controls.
- Silent inputs remain one fixed embedding per modality, so this tests recovery of an
  ordered utterance signature rather than frame-to-frame sensor alignment.

## Per-Fold Results

| Fold | Temporal Student Accuracy | Pooled Student Accuracy | True-Order Segment Cosine | Reversed Cosine | Order Margin |
|---:|---:|---:|---:|---:|---:|
| 0 | 65.8% | 67.5% | 0.361 | 0.112 | +0.250 |
| 1 | 57.8% | 61.1% | 0.293 | 0.061 | +0.232 |
| 2 | 65.7% | 62.7% | 0.369 | 0.094 | +0.274 |
| 3 | 78.3% | 74.8% | 0.347 | 0.070 | +0.277 |
| 4 | 54.8% | 53.9% | 0.357 | 0.091 | +0.267 |

## Aggregate

- Temporal-student class accuracy: **64.5% +/- 9.1%**.
- Pooled-HuBERT student accuracy: **64.0% +/- 7.8%**.
- True-order segment cosine: **0.346**.
- Reversed-order cosine: **0.086**.
- Shifted-order cosine: **0.070**.
- Mean true-versus-reversed order margin: **+0.260**.
- Train-mean target baseline cosine: **0.044**.
- Segment-position cosines: S1 0.324, S2 0.394, S3 0.315, S4 0.349.

The positive true-order margin indicates that the student recovers some ordered speech structure, not only an order-invariant utterance summary.

## Boundary

Relative-time targets are a stricter teacher than one global mean, but they do not create
framewise silent-sensor observations. Phoneme-level claims require either temporal sensor
encoder activations or explicit forced-alignment/articulatory labels.
