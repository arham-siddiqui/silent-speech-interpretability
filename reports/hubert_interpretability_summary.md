# HuBERT Student Interpretability Summary

This report consolidates the first real audio-teacher interpretability batch for the
contactless / microphone-free student.

![Probe accuracy](figures/hubert_student_probe_accuracy.svg)

![Modality attribution](figures/hubert_modality_attribution.svg)

## Core Results

- Five-fold student accuracy: **64.0%**, compared
  with **63.9%** for strict validation-weighted fusion.
- Residual-HuBERT cosine: **0.430**, versus
  **-0.001** for the train-mean residual direction.
- Bottleneck class probe: **64.9%**.
- Bottleneck utterance-type probe: **95.1%**.
- Speaker leakage falls from **61.4%**
  at sensor input to **11.5%**
  at the bottleneck.
- Lip alone reaches **61.0%**; removing lip drops
  accuracy by **27.2 points**.

## Interpretation

The 64-dimensional bottleneck preserves utterance class and coarse speech-type
information while removing most linearly decodable speaker identity. Lip is largely
sufficient for class decoding, while laser provides the largest auxiliary leave-one-out
gain. UWB and mmWave have measurable standalone information but little conditional
accuracy contribution once lip and the other sensors are present.

Centering HuBERT targets with training-fold statistics was essential. Without centering,
a trivial shared mean direction achieved very high cosine similarity and obscured
utterance-varying alignment. All final results use centered targets without test-speaker
statistics.

## Limits And Next Step

Linear probes establish decodability, and modality retraining establishes attribution at
the input level; neither proves that individual bottleneck features causally control a
speech property. The next phase should train a sparse autoencoder on the bottleneck,
rank features by class/type selectivity, then ablate the top features across held-out
speakers.
