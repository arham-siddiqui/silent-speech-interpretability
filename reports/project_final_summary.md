# Silent Speech Interpretability: Consolidated Findings

## Project objective

Build an interpretable silent-speech system from RVTALL non-audio sensors, establish a
strict speaker/encoder-disjoint supervised baseline, and use audio teachers to determine
which speech and articulatory information survives in silent-sensor representations.

## Completed evidence

| Track | Evaluation | Main result | Status |
|---|---|---:|---|
| Supervised baseline | Five-fold true encoder-disjoint RVTALL CV | 63.9% validation-weighted fusion | Complete |
| Audio teacher/student | Speaker-disjoint HuBERT student | 64.0% pooled student accuracy | Complete |
| Ordered representation | Four-segment silent-sensor to HuBERT transfer | 0.381 true-order cosine | Complete |
| Multitask temporal model | Speaker-disjoint class and HuBERT prediction | 60.1% accuracy; 0.386 cosine | Complete |
| Phonetic interpretation | Strict fold-valid phone occupancy probes | +0.038 all-sensor residual R2 | Complete |
| Manual timing audit | 40 balanced recordings | Exact timing trails matched uniform by -0.0025 R2 | Complete |
| External session replication | Three seeds, held-out radar sessions | 28.7% accuracy; 0.327 cosine | Complete |
| External speaker transfer | Three seeds, leave-one-subject-out radar | 3.0% accuracy; -0.012 order margin | Negative result |

## Supported conclusions

1. **The supervised baseline is reliable.** Validation-weighted late fusion reaches
   63.9% under true speaker- and encoder-disjoint CV. Mouth remains excluded from fusion
   because fold-specific artifacts stay near chance.
2. **Silent sensors predict useful speech-teacher structure.** HuBERT-aligned students
   retain class and coarse ordered information under speaker-disjoint RVTALL evaluation.
3. **The phonetic signal is coarse rather than frame-exact.** Contactless modalities add
   residual phonetic occupancy information, but exact phone boundaries do not outperform
   matched uniform timing. The manual audit confirms this boundary.
4. **The method replicates across recording sessions in an independent radar corpus.**
   Multi-seed accuracy is **28.7%**
   [19.9%,
   37.0%] versus 2% chance. The
   within-command residual order margin remains **+0.122**.
5. **The current external radar representation is not speaker-independent.**
   Leave-one-subject-out accuracy is only **3.0%**
   [2.6%,
   3.4%], and its temporal order
   interval includes zero.

## Claims not supported

- Broad external speaker generalization from radar.
- Exact phone tracking from the current four-segment representations.
- One-feature/one-phoneme selectivity.
- A conclusion that adding architectural attention alone improves interpretability.

## Technical interpretation

The project succeeds at strict RVTALL decoding and at identifying coarse, ordered
speech-related structure in silent sensors. External ablations show that S32 and
magnitude features carry most session-transfer performance, while S12 and temporal
differences are weaker alone. The collapse under unseen-subject transfer indicates that
speaker-specific radar response and geometry remain dominant nuisance variables.

## Highest-value continuation

1. Add speaker-invariant radar normalization and source-subject domain adversarial
   training, selected without target-subject validation.
2. Evaluate on a larger external speaker cohort; two subjects cannot support a
   population-level generalization claim.
3. Preserve the current interpretability boundary and avoid further exact-phone or
   neuron-selectivity claims without higher-resolution synchronized targets.

## Evidence map

- [`true_encoder_cv_results.md`](true_encoder_cv_results.md)
- [`temporal_sensor_interpretability.md`](temporal_sensor_interpretability.md)
- [`phone_ctc_interpretability.md`](phone_ctc_interpretability.md)
- [`phone_boundary_audit_results.md`](phone_boundary_audit_results.md)
- [`external_radar_hubert_replication.md`](external_radar_hubert_replication.md)
- [`external_radar_generalization.md`](external_radar_generalization.md)
