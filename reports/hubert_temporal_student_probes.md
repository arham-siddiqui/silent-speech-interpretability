# HuBERT Student Interpretability Probes

Frozen linear probes were evaluated across the same five encoder-disjoint folds as the
student CV experiment.

## Probe Design

- Utterance class and type probes use speaker-disjoint train/validation/test splits.
- Speaker leakage probes use training speakers in every split but hold out complete
  utterance classes, testing whether identity generalizes across unseen content.
- Probe regularization is selected on validation data before refitting on train plus
  validation data.
- `Teacher HuBERT` is the real audio target and serves as a reference, not a sensor-only
  inference representation.

## Aggregate Results

| Task | Representation | Mean Accuracy | Std. Dev. | Chance |
|---|---|---:|---:|---:|
| Speaker Leakage | Bottleneck | 10.4% | 3.1% | 7.5% |
| Speaker Leakage | Hidden | 25.5% | 2.5% | 7.5% |
| Speaker Leakage | Predicted Hubert | 12.5% | 2.9% | 7.5% |
| Speaker Leakage | Sensor Input | 61.4% | 4.2% | 7.5% |
| Speaker Leakage | Teacher Hubert | 88.6% | 3.6% | 7.5% |
| Utterance Class | Bottleneck | 65.8% | 7.6% | 3.3% |
| Utterance Class | Hidden | 65.0% | 8.1% | 3.3% |
| Utterance Class | Predicted Hubert | 63.0% | 9.7% | 3.3% |
| Utterance Class | Sensor Input | 67.6% | 9.3% | 3.3% |
| Utterance Class | Teacher Hubert | 81.7% | 4.5% | 3.3% |
| Utterance Type | Bottleneck | 94.3% | 2.1% | 33.3% |
| Utterance Type | Hidden | 94.5% | 2.8% | 33.3% |
| Utterance Type | Predicted Hubert | 94.6% | 1.5% | 33.3% |
| Utterance Type | Sensor Input | 94.1% | 2.6% | 33.3% |
| Utterance Type | Teacher Hubert | 95.7% | 2.5% | 33.3% |

## Main Findings

- The strongest class representation is **teacher hubert** at
  **81.7%** mean speaker-disjoint accuracy.
- The strongest utterance-type representation is **teacher hubert**
  at **95.7%**.
- Speaker leakage changes from **61.4%**
  in the concatenated sensor input to **10.4%**
  in the student bottleneck. Higher speaker-probe accuracy means more identity leakage.

These probes measure linear decodability, not causal use. The modality attribution
experiment and later feature ablations are needed to determine which sensor inputs and
features drive the decoded information.
