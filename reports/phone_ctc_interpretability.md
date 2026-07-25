# Direct Phone-CTC Interpretability

## Alignment Audit

The direct IPA phoneme recognizer aligned **596 / 596**
paired recordings and produced **4854** canonical phone intervals. Only
**34** word spans required a marked uniform fallback after phone emissions
collapsed at a word anchor. The primary probe uses **178 / 198** sentence recordings across
all 20 speakers and all 10 sentence classes; short isolated words are excluded because
their phone-recognition quality is not balanced across classes.

## Speaker-Disjoint Sentence Probes

| Representation | R2 | Class + position baseline | Delta R2 | Order margin |
|---|---:|---:|---:|---:|
| all_modalities | 0.428 | 0.396 | +0.032 | +0.352 |
| contactless_nonlip | 0.425 | 0.396 | +0.029 | +0.346 |
| attention_temporal_student | 0.416 | 0.396 | +0.020 | +0.345 |
| laser | 0.426 | 0.407 | +0.019 | +0.319 |
| uwb | 0.425 | 0.407 | +0.018 | +0.326 |
| multitask_temporal_student | 0.411 | 0.396 | +0.015 | +0.342 |
| mmwave | 0.406 | 0.396 | +0.009 | +0.306 |
| lip | 0.399 | 0.407 | -0.008 | +0.314 |

The strongest primary result is all modalities at **+0.032**
residual R2, followed by contactless non-lip sensors at
**+0.029**. Lip alone does not improve the
class/position baseline (**-0.008**).

## Quality Sensitivity

| Gate | Sentences | All modalities Delta R2 | Contactless Delta R2 |
|---|---:|---:|---:|
| direct_lenient | 191 | +0.029 | +0.025 |
| direct_main | 178 | +0.032 | +0.029 |
| direct_strict_cv | 157 | +0.038 | +0.034 |

The positive result strengthens under the strictest fold-valid gate, so it is not driven
by low-confidence alignments.

## Exact-Boundary Control

Using uniform phone subdivisions on the exact same 178 sentences gives
**+0.034** all-modality residual R2 versus
**+0.032** with direct phone timing. The mean
direct-minus-uniform gain is **-0.002** across representations.
Thus the experiment supports broad ordered phonetic occupancy, but does not show that the
sensor representations track the sharper within-word phone boundaries.

## Best Feature Families

| Phonetic family | Best representation | Delta R2 | Order margin |
|---|---|---:|---:|
| affricate | attention_temporal_student | +0.036 | +0.229 |
| fricative | uwb | +0.012 | +0.310 |
| glide | all_modalities | +0.040 | +0.694 |
| liquid | all_modalities | +0.036 | +0.524 |
| nasal | multitask_temporal_student | +0.011 | +0.196 |
| silence | contactless_nonlip | +0.059 | +0.265 |
| stop | all_modalities | +0.046 | +0.316 |
| vowel | contactless_nonlip | +0.057 | +0.269 |

## Sparse-Feature Linkage

The fold-local temporal HuBERT sparse codes add **+0.010**
macro R2 over class/position. Ablating the 50 phone-ranked features loses
**0.0033** R2 versus **0.0010** for random
features, but only **1 / 5** folds reach a within-fold
random-control p-value below 0.05. Their mean top-50 overlap with existing HuBERT content
features is **3.2**, below the chance expectation of
**4.9**, and foldwise rank correlation averages **-0.139**.

This is a controlled negative: phone timing is weakly recoverable from the sparse code but
is not stably concentrated in the previously identified HuBERT-causal content features.
No individual sparse feature should be named as a phoneme from these results.
