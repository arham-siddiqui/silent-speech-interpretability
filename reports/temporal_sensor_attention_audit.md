# Temporal Modality-Attention Audit

This audit summarizes learned weights on held-out speakers only. The attention model is
diagnostic rather than the selected model because it underperformed the simpler multitask
student on both classification and HuBERT alignment.

## Classification Temporal Attention

Each row sums to one across the four relative-time segments.

| Modality | S1 | S2 | S3 | S4 |
|---|---:|---:|---:|---:|
| Lip | 0.077 | 0.311 | 0.457 | 0.155 |
| Laser | 0.242 | 0.204 | 0.240 | 0.314 |
| Mmwave | 0.264 | 0.224 | 0.236 | 0.276 |
| Uwb | 0.130 | 0.340 | 0.338 | 0.192 |

## HuBERT Modality Fusion

Each row sums to one across modalities.

| Segment | Lip | Laser | Mmwave | Uwb |
|---|---:|---:|---:|---:|
| S1 | 0.281 | 0.159 | 0.215 | 0.346 |
| S2 | 0.224 | 0.315 | 0.287 | 0.174 |
| S3 | 0.282 | 0.294 | 0.178 | 0.246 |
| S4 | 0.311 | 0.216 | 0.263 | 0.210 |

## Concentration

- Normalized temporal-attention entropy: **0.880** (`1.0` is uniform).
- Normalized modality-fusion entropy: **0.917** (`1.0` is uniform).

High entropy indicates diffuse weighting rather than a sharp sensor/time selection. These
weights describe the learned attention model; they are not causal modality importance.
