# External Radar Generalization And Ablation Batch

## Protocols

- Session-held-out: one session trains, one validates, and one tests, with both
  subjects represented in each split.
- Subject-held-out: two sessions from one subject train, that subject's third session
  validates, and all sessions from the other subject test.
- Reliability: 3 optimization seeds per held-out unit.
- Intervals: empirical 95% bootstrap intervals over held-out sessions or subjects after
  averaging seeds. With only three sessions and two subjects, these intervals describe
  this corpus and are not population-level confidence intervals.

## Multi-seed results

| Protocol | Accuracy (95% interval) | HuBERT cosine | Order margin | Residual cosine | Residual order margin |
|---|---:|---:|---:|---:|---:|
| Session-held-out | 28.7% [19.9%, 37.0%] | 0.327 [0.260, 0.363] | 0.339 [0.262, 0.386] | 0.154 [0.128, 0.169] | 0.122 [0.104, 0.136] |
| Subject-held-out | 3.0% [2.6%, 3.4%] | -0.014 [-0.023, -0.004] | -0.012 [-0.028, 0.004] | 0.030 [0.020, 0.041] | 0.005 [-0.011, 0.020] |

Chance command accuracy is 2%.

## Optimization stability

- Session-held-out mean within-session seed SD: **2.5 accuracy points** and **0.009 cosine**.
- Subject-held-out mean within-subject seed SD: **0.3 accuracy points** and **0.018 cosine**.

## Subject transfer detail

| Held-out subject | Accuracy | HuBERT cosine | Order margin | Residual cosine | Residual order margin |
|---|---:|---:|---:|---:|---:|
| S001 | 2.6% | -0.004 | +0.004 | 0.041 | +0.020 |
| S002 | 3.4% | -0.023 | -0.028 | 0.020 | -0.011 |

## Radar feature ablations

Each ablation uses the established session-held-out protocol and seed 42. S12 and S32
denote the two scattering-parameter radargrams used by the source implementation.

| Feature set | Accuracy | HuBERT cosine | Order margin |
|---|---:|---:|---:|
| s32 | 28.2% | 0.282 | +0.287 |
| all | 27.8% | 0.323 | +0.330 |
| magnitude | 27.1% | 0.258 | +0.278 |
| delta | 16.6% | 0.176 | +0.158 |
| s12 | 14.5% | 0.189 | +0.195 |

## Interpretation

Session transfer measures robustness to a new recording visit with familiar subjects.
Subject transfer is the stricter speaker-generalization test. The residual metrics remove
train-only command prototypes, so they test ordered exemplar variation beyond command
identity. Feature ablations identify whether performance depends more on radar channel
choice or on static magnitude versus temporal-change information.

The external corpus contains only two subjects, so its held-out-subject result should
be treated as a diagnostic boundary rather than a population-level estimate.
