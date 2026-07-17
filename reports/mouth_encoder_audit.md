# Mouth Encoder Audit

The true encoder-disjoint CV run shows mouth embeddings near chance:
mean accuracy 5.7% +/- 0.9%
across 5 folds.

## Fold Results

| Fold | Accuracy | Macro F1 | Test Samples |
| --- | --- | --- | --- |
| 0 | 5.1% | 2.2% | 117 |
| 1 | 4.4% | 2.0% | 90 |
| 2 | 6.9% | 2.8% | 102 |
| 3 | 6.1% | 6.2% | 115 |
| 4 | 6.1% | 3.5% | 115 |

## Artifact Metadata Notes

| Fold | Embedding Path | Max Epochs | Note |
| --- | --- | --- | --- |
| 0 | artifacts/embeddings/speaker_cv/fold_0/mouth_embeddings.npz | 60 | Short max_epochs values are smoke tests, not final scientific fold embeddings. |
| 1 | artifacts/embeddings/speaker_cv/fold_1/mouth_embeddings.npz | 60 | Short max_epochs values are smoke tests, not final scientific fold embeddings. |
| 2 | artifacts/embeddings/speaker_cv/fold_2/mouth_embeddings.npz | 60 | Short max_epochs values are smoke tests, not final scientific fold embeddings. |
| 3 | artifacts/embeddings/speaker_cv/fold_3/mouth_embeddings.npz | 60 | Short max_epochs values are smoke tests, not final scientific fold embeddings. |
| 4 | artifacts/embeddings/speaker_cv/fold_4/mouth_embeddings.npz | 60 | Short max_epochs values are smoke tests, not final scientific fold embeddings. |

## Interpretation

The mouth artifacts are present and pass the artifact contract, but their metadata marks
them as projection/smoke-test style artifacts rather than final scientific mouth encoder
folds. The near-chance CV accuracy is therefore best interpreted as an artifact-quality
issue, not evidence that mouth video lacks signal.

## Recommended Fix

Retrain fold-specific mouth video embeddings with the same encoder-disjoint discipline as
the lip, laser, UWB, and mmWave artifacts, then rerun `scripts/08_run_true_encoder_cv.py`
and regenerate this report.
