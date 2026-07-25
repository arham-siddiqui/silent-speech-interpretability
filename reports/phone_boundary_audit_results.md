# Manual Phone Boundary Audit

## Review Outcome

All **40 / 40** sampled sentence recordings received a listening
decision across all 20 speakers and all 10 sentence classes:

- Accepted unchanged: **29**
- Corrected: **5**
- Excluded as unusable alignments: **6**
- Median nonzero correction: **97.8 ms**
- Maximum correction: **193.4 ms**

| Automatic quality bin | Accepted | Corrected | Excluded |
|---|---:|---:|---:|
| borderline | 2 | 0 | 1 |
| fallback | 1 | 1 | 2 |
| high | 22 | 3 | 0 |
| primary | 4 | 1 | 3 |

None of the 25 high-confidence examples was excluded. All six exclusions came from the
primary, borderline, or fallback strata. The automated quality diagnostics are therefore
informative, but the three exclusions in the primary stratum show that they do not replace
manual review.

## Speaker-Disjoint Probe Comparison

The primary set retains **175** quality-controlled sentences after audit. **Correction
gain** compares corrected and original boundaries on these exact 175 recordings.
**Boundary gain** compares corrected boundaries with uniform within-word timing on the
same recordings.

| Representation | Original Delta R2 | Audited Delta R2 | Correction gain | Uniform Delta R2 | Boundary gain |
|---|---:|---:|---:|---:|---:|
| all_modalities | +0.0321 | +0.0337 | +0.0000 | +0.0353 | -0.0015 |
| attention_temporal_student | +0.0195 | +0.0213 | +0.0003 | +0.0232 | -0.0019 |
| contactless_nonlip | +0.0286 | +0.0303 | +0.0000 | +0.0313 | -0.0010 |
| laser | +0.0185 | +0.0187 | -0.0003 | +0.0221 | -0.0034 |
| lip | -0.0081 | -0.0071 | -0.0000 | +0.0025 | -0.0096 |
| mmwave | +0.0091 | +0.0091 | -0.0002 | +0.0107 | -0.0016 |
| multitask_temporal_student | +0.0151 | +0.0163 | -0.0002 | +0.0175 | -0.0012 |
| uwb | +0.0178 | +0.0177 | +0.0004 | +0.0173 | +0.0004 |

Across representations, manual correction changes residual R2 by
**+0.0000** on average. Audited boundaries trail matched
uniform timing by **-0.0025** on average.
All modalities remain positive at
**+0.0337**, and non-lip contactless
sensors remain positive at
**+0.0303**.

## Interpretation

Manual review confirms that the positive phonetic result is not driven by obviously bad
alignments: excluding six failures and correcting five recordings leaves the signal
unchanged. It also confirms the controlled negative for exact timing. The representations
carry broad ordered phonetic occupancy beyond sentence class and relative position, but
the evidence does not show sensitivity to sharper within-word phone transitions.

The defensible project claim remains **coarse phonetic progression, not exact phone
tracking or individual phoneme neurons**. Phone identities are canonical prompt
pronunciations rather than manually transcribed speaker realizations. The next scientific
step is external-cohort validation, not additional tuning on these 40 audit examples.
