# HuBERT Student Modality Attribution

Each student variant was trained across all five encoder-disjoint folds against the
same HuBERT targets. Every variant uses the full four-modality pair intersection, so
sample coverage and held-out speakers are identical across comparisons.

## Results

| Variant | Included Modalities | Mean Accuracy | Std. Dev. | Delta vs Full (pp) | Target Cosine | Target MSE |
|---|---|---:|---:|---:|---:|---:|
| Full | lip, uwb, mmwave, laser | 64.0% | 7.8% | +0.0 | 0.430 | 1.140 |
| Single Lip | lip | 61.0% | 6.1% | -3.0 | 0.410 | 1.180 |
| Single Uwb | uwb | 28.1% | 10.3% | -35.9 | 0.350 | 1.299 |
| Single Laser | laser | 21.3% | 4.1% | -42.8 | 0.372 | 1.255 |
| Single Mmwave | mmwave | 16.2% | 4.7% | -47.8 | 0.363 | 1.274 |
| Leave Out Uwb | lip, mmwave, laser | 64.9% | 9.6% | +0.9 | 0.427 | 1.147 |
| Leave Out Mmwave | lip, uwb, laser | 63.9% | 9.3% | -0.1 | 0.420 | 1.161 |
| Leave Out Laser | lip, uwb, mmwave | 62.3% | 5.3% | -1.7 | 0.419 | 1.162 |
| Leave Out Lip | uwb, mmwave, laser | 36.8% | 7.7% | -27.2 | 0.391 | 1.218 |

## Main Findings

- The full four-sensor student reaches **64.0%** mean accuracy.
- The strongest single-sensor student is **lip**
  at **61.0%**.
- Removing **lip** produces the largest mean accuracy drop relative to the full
  student (**-27.2 percentage points**).

Single-modality performance measures sufficiency, while leave-one-out changes measure
conditional contribution given the other sensors. Neither establishes causal feature
mechanisms inside the network; that requires activation-level ablation.

The train-mean HuBERT direction has mean cosine similarity
**-0.001** on held-out speakers. Student target
cosine should be interpreted relative to this baseline; class accuracy and target
alignment need not move together because the training objective contains both losses.
