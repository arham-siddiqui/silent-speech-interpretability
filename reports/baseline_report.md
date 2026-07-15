# Silent Speech Interpretability Baseline Report

This report summarizes the current contactless / microphone-free speech decoding baseline.

## Dataset Audit

- Manifest samples: 599
- Strict five-modality intersection: 539
- Embedding strict intersection: 539
- Label mismatches: 0
- User ID mismatches: 0
- Embedding sources: {'laser': 'configured', 'lip': 'configured', 'mmwave': 'configured', 'mouth': 'configured', 'uwb': 'configured'}

## Fixed Speaker Split

| method               | modality | num_train | num_test | accuracy | macro_f1 |
| -------------------- | -------- | --------- | -------- | -------- | -------- |
| prototype            | lip      | 4254      | 60       | 0.483    | 0.442    |
| prototype            | mouth    | 4375      | 60       | 0.500    | 0.488    |
| prototype            | uwb      | 4175      | 60       | 0.300    | 0.243    |
| prototype            | mmwave   | 3876      | 60       | 0.383    | 0.337    |
| prototype            | laser    | 4015      | 60       | 0.467    | 0.408    |
| equal_weight         | fusion   | 420       | 60       | 0.750    | 0.735    |
| borda                | fusion   | 420       | 60       | 0.767    | 0.746    |
| consistency_weighted | fusion   | 420       | 60       | 0.767    | 0.751    |

Best fixed-split method: `borda` at accuracy 0.767.

## Legacy-Compatible Comparison

| baseline_family   | method                  | num_train | num_val | num_test | accuracy | macro_f1 |
| ----------------- | ----------------------- | --------- | ------- | -------- | -------- | -------- |
| auditable_current | equal_weight            | 420       | 59      | 60       | 0.750    | 0.735    |
| auditable_current | borda                   | 420       | 59      | 60       | 0.767    | 0.746    |
| auditable_current | consistency_weighted    | 420       | 59      | 60       | 0.767    | 0.751    |
| legacy_compatible | legacy_equal_weight     | 3509      | 59      | 60       | 0.733    | 0.699    |
| legacy_compatible | legacy_borda            | 3509      | 59      | 60       | 0.783    | 0.773    |
| legacy_compatible | legacy_hard_consistency | 3509      | 59      | 60       | 0.783    | 0.765    |

## 5-Fold Speaker-Disjoint CV

Sanity note: 0/5 CV folds are encoder-disjoint for these precomputed embeddings. Treat this table as fusion-layer CV unless encoders are retrained inside each fold.

| method               | modality | mean  | std   | count |
| -------------------- | -------- | ----- | ----- | ----- |
| borda                | fusion   | 0.922 | 0.040 | 5     |
| consistency_weighted | fusion   | 0.931 | 0.045 | 5     |
| equal_weight         | fusion   | 0.924 | 0.047 | 5     |
| prototype            | laser    | 0.416 | 0.070 | 5     |
| prototype            | lip      | 0.832 | 0.102 | 5     |
| prototype            | mmwave   | 0.506 | 0.034 | 5     |
| prototype            | mouth    | 0.568 | 0.045 | 5     |
| prototype            | uwb      | 0.827 | 0.095 | 5     |

Best CV method: `consistency_weighted` / `fusion` at mean accuracy 0.931.

## True Encoder-Disjoint CV Status

True encoder-disjoint CV is prepared but not yet runnable: 25 fold-specific embedding artifacts are missing or invalid. See `reports/true_encoder_cv_plan.md` and `reports/results/true_encoder_cv_missing_artifacts.csv`.

## Figures

- `reports/figures/generated/fixed_split_accuracy_bar.png`
- `reports/figures/generated/confusion_matrix_fixed_split.png`
- `reports/figures/generated/speaker_cv_accuracy.png`
- `reports/figures/generated/speaker_cv_by_modality.png`

## Notes

- Audio is not used in this baseline inference path.
- Fusion metrics use the strict multimodal intersection.
- Individual modality fixed-split metrics use each modality's available test pairs.
- Current CV uses precomputed embeddings, so it is not full encoder-disjoint CV unless encoders are retrained per fold.
