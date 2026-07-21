# Temporal Phonetic Probes

## Result

The strongest macro-average residual probe is **contactless_nonlip**, with R2
**0.511** versus the utterance-class/position baseline **0.500**
(delta **+0.011**). The residual design asks whether a representation explains
speaker-specific timing after the expected phonetic trajectory for each class is removed.

| Representation | R2 | Class + position baseline | Delta R2 | Correlation | Order margin |
|---|---:|---:|---:|---:|---:|
| contactless_nonlip | 0.511 | 0.500 | +0.011 | 0.717 | +0.167 |
| all_modalities | 0.510 | 0.500 | +0.009 | 0.716 | +0.172 |
| uwb | 0.523 | 0.515 | +0.008 | 0.724 | +0.179 |
| attention_temporal_student | 0.507 | 0.500 | +0.007 | 0.717 | +0.177 |
| laser | 0.516 | 0.511 | +0.005 | 0.721 | +0.166 |
| multitask_temporal_student | 0.504 | 0.500 | +0.004 | 0.715 | +0.177 |
| mmwave | 0.506 | 0.504 | +0.002 | 0.712 | +0.152 |
| lip | 0.515 | 0.515 | -0.000 | 0.720 | +0.172 |

## Best Representation Per Feature

| Feature | Representation | R2 | Baseline R2 | Delta R2 | Correlation |
|---|---|---:|---:|---:|---:|
| affricate | uwb | 0.348 | 0.346 | +0.002 | 0.622 |
| fricative | contactless_nonlip | 0.416 | 0.396 | +0.019 | 0.660 |
| glide | attention_temporal_student | 0.357 | 0.355 | +0.002 | 0.618 |
| liquid | all_modalities | 0.403 | 0.378 | +0.025 | 0.649 |
| nasal | uwb | 0.488 | 0.486 | +0.002 | 0.708 |
| silence | contactless_nonlip | 0.790 | 0.768 | +0.022 | 0.891 |
| stop | contactless_nonlip | 0.409 | 0.372 | +0.037 | 0.659 |
| vowel | contactless_nonlip | 0.959 | 0.955 | +0.004 | 0.979 |

## Interpretation Boundary

These targets combine CTC-aligned word boundaries with uniformly interpolated ARPAbet
phones. Results support broad, time-varying phonetic **occupancy** only when they improve
over the class/position baseline and retain temporal order. They do not establish exact
phone boundaries or a one-neuron/one-phoneme correspondence. The main analysis excludes
word alignments below confidence 0.05 and retains known isolated-vowel intervals.
