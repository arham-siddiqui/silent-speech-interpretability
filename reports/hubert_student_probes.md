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
| Speaker Leakage | Bottleneck | 11.5% | 2.9% | 7.5% |
| Speaker Leakage | Hidden | 30.0% | 5.0% | 7.5% |
| Speaker Leakage | Predicted Hubert | 15.1% | 3.9% | 7.5% |
| Speaker Leakage | Sensor Input | 61.4% | 4.2% | 7.5% |
| Speaker Leakage | Teacher Hubert | 88.3% | 4.8% | 7.5% |
| Utterance Class | Bottleneck | 64.9% | 8.7% | 3.3% |
| Utterance Class | Hidden | 66.1% | 7.7% | 3.3% |
| Utterance Class | Predicted Hubert | 61.9% | 11.0% | 3.3% |
| Utterance Class | Sensor Input | 67.6% | 9.3% | 3.3% |
| Utterance Class | Teacher Hubert | 87.3% | 4.9% | 3.3% |
| Utterance Type | Bottleneck | 95.1% | 2.8% | 33.3% |
| Utterance Type | Hidden | 96.8% | 1.6% | 33.3% |
| Utterance Type | Predicted Hubert | 94.8% | 2.4% | 33.3% |
| Utterance Type | Sensor Input | 94.1% | 2.6% | 33.3% |
| Utterance Type | Teacher Hubert | 96.3% | 2.6% | 33.3% |

## Main Findings

- The strongest class representation is **teacher hubert** at
  **87.3%** mean speaker-disjoint accuracy.
- The strongest utterance-type representation is **hidden**
  at **96.8%**.
- Speaker leakage changes from **61.4%**
  in the concatenated sensor input to **11.5%**
  in the student bottleneck. Higher speaker-probe accuracy means more identity leakage.

These probes measure linear decodability, not causal use. The modality attribution
experiment and later feature ablations are needed to determine which sensor inputs and
features drive the decoded information.
