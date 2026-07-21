# Audio Teacher Comparison

HuBERT and Wav2Vec2 were compared with the same four-segment pooling, silent-sensor
activations, student architecture, optimization, and five encoder/speaker-disjoint folds.

| Teacher | Accuracy | Segment cosine | Reversed cosine | Order margin | Target MSE |
|---|---:|---:|---:|---:|---:|
| HuBERT | 49.9% | 0.381 | 0.047 | +0.333 | 1.239 |
| Wav2Vec2 | 47.9% | 0.290 | 0.139 | +0.150 | 1.421 |

Wav2Vec2 changes accuracy by **-2.0 percentage points** and true-order
cosine by **-0.091**. Because it is worse on
both objectives and has a substantially smaller order margin, HuBERT remains the selected
audio teacher. A Wav2Vec2 multitask hyperparameter sweep is not promoted: the matched base
experiment already rejects the hypothesis that this alternate teacher improves temporal
transfer under the current architecture.
