# True Encoder-Disjoint CV Results

This report summarizes the completed real-encoder, speaker/encoder-disjoint
cross-validation run for the silent speech interpretability project.

## Evaluation Setup

- Dataset: RVTALL silent speech subset, 30 classes.
- Modalities: lip landmarks, mouth video, UWB radar, mmWave radar, laser speckle.
- Split: 5-fold encoder-disjoint speaker CV.
- Classifier: prototype/nearest-centroid per modality, plus late-fusion voting.
- Fusion methods: equal-weight probability averaging, Borda rank fusion, and
  consistency-weighted fusion.
- Chance accuracy: 3.3%.
- All reported folds have `encoder_disjoint_test=True`.

The raw outputs are saved in:

- `reports/results/true_encoder_cv_results.csv`
- `reports/results/true_encoder_cv_summary.csv`
- `reports/results/true_encoder_cv_missing_artifacts.csv`

`true_encoder_cv_missing_artifacts.csv` is intentionally empty except for the header,
meaning all required fold/modality artifacts are present.

## Aggregate Accuracy

| Method | Modality | Mean Accuracy | Std. Dev. | Folds |
|---|---:|---:|---:|---:|
| Equal-weight fusion | fusion | 61.7% | 7.9% | 5 |
| Prototype | lip | 60.9% | 6.6% | 5 |
| Consistency-weighted fusion | fusion | 58.5% | 8.5% | 5 |
| Borda fusion | fusion | 40.1% | 10.2% | 5 |
| Prototype | UWB | 26.7% | 9.3% | 5 |
| Prototype | laser | 24.3% | 5.5% | 5 |
| Prototype | mmWave | 15.7% | 1.2% | 5 |
| Prototype | mouth | 5.7% | 0.9% | 5 |

## Per-Fold Accuracy

| Fold | Lip | Mouth | UWB | mmWave | Laser | Equal Fusion | Borda | Consistency Fusion |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 61.5% | 5.1% | 27.4% | 13.7% | 16.2% | 62.4% | 33.3% | 57.3% |
| 1 | 56.7% | 4.4% | 14.4% | 16.7% | 28.9% | 55.6% | 28.9% | 50.0% |
| 2 | 56.9% | 6.9% | 33.3% | 16.7% | 21.6% | 62.7% | 38.2% | 59.8% |
| 3 | 72.2% | 6.1% | 37.4% | 15.7% | 29.6% | 73.9% | 54.8% | 72.2% |
| 4 | 57.4% | 6.1% | 20.9% | 15.7% | 25.2% | 53.9% | 45.2% | 53.0% |

## Main Takeaways

1. Lip embeddings are the strongest single modality in the real encoder-disjoint
   evaluation, averaging 60.9% accuracy.
2. Equal-weight fusion is the strongest overall method, averaging 61.7% accuracy.
   It only modestly improves over lip alone, so most of the reliable signal is still
   coming from the lip encoder.
3. UWB and laser carry useful auxiliary signal, but they are weaker and more variable
   than lip.
4. mmWave is consistently above chance but much weaker than expected from the earlier
   non-CV baseline.
5. Mouth embeddings are near chance in this completed CV setting and should not be
   trusted until their artifact path/training setup is re-audited.
6. Consistency-weighted fusion underperforms equal-weight fusion on average, which
   suggests the current agreement heuristic is suppressing useful lip signal or giving
   too much influence to weak modalities.

## Verification

The final run completed with:

```text
pytest -q
19 passed
```

Strict true encoder CV also completed without `--allow-missing`, confirming that the
full artifact contract is satisfied.

## Recommended Next Steps

1. Add a reliability-aware fusion method that learns fold-level or validation-derived
   modality weights instead of weighting every modality equally.
2. Run a per-class error analysis to find which utterance classes benefit from UWB,
   laser, and mmWave versus which classes are mostly lip-driven.
3. Audit the mouth encoder path, because mouth performance is unexpectedly near chance
   in this CV run.
4. Add a small reporting script that regenerates this Markdown report directly from
   `true_encoder_cv_results.csv` and `true_encoder_cv_summary.csv`.
5. Once the fusion/reporting scripts are stable, update the README's older fixed-split
   numbers so it clearly distinguishes legacy baseline results from the stricter
   true encoder-disjoint CV results.
