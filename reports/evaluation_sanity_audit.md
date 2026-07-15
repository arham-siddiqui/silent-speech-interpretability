# Evaluation Sanity Audit

This audit checks whether evaluation splits are disjoint from the speakers used to train the precomputed encoders.

- Encoder training speakers: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
- Fixed test speakers: [19, 20]
- Fixed split encoder-disjoint: True
- Encoder-disjoint CV folds: 0/5

## CV Fold Overlap

| fold | test_speakers | encoder_seen_test_speakers | num_encoder_seen_test_speakers | encoder_disjoint_test |
| ---- | ------------- | -------------------------- | ------------------------------ | --------------------- |
| 0    | 8,10,15,16    | 8,10,15,16                 | 4                              | False                 |
| 1    | 7,11,13,20    | 7,11,13                    | 3                              | False                 |
| 2    | 1,4,6,17      | 1,4,6                      | 3                              | False                 |
| 3    | 3,5,12,19     | 3,5,12                     | 3                              | False                 |
| 4    | 2,9,14,18     | 2,9,14                     | 3                              | False                 |

## Interpretation

CV over precomputed embeddings is fusion-layer CV only when held-out speakers were used to train the encoders. Treat fixed split as the encoder-disjoint baseline unless encoders are retrained per fold.
