# True Encoder-Disjoint CV Preparation Plan

Each fold needs encoders trained only on that fold's train speakers, then embeddings extracted for all speakers using those fold-specific encoders.

Expected artifacts per fold:

## Fold 0

- Train speakers: [7, 20, 4, 1, 17, 6, 12, 19, 3, 5, 18, 2, 14, 9]
- Val speakers: [13, 11]
- Test speakers: [16, 10, 15, 8]
- Metadata: `artifacts/embeddings/speaker_cv/fold_0/metadata.json`
- Lip command:
  `python3 scripts/09_train_lip_fold_embeddings.py --config configs/real_embeddings.local.yaml --fold 0`

- `artifacts/embeddings/speaker_cv/fold_0/lip_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_0/mouth_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_0/uwb_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_0/mmwave_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_0/laser_embeddings.npz`

## Fold 1

- Train speakers: [15, 8, 4, 1, 17, 6, 12, 19, 3, 5, 18, 2, 14, 9]
- Val speakers: [16, 10]
- Test speakers: [13, 11, 7, 20]
- Metadata: `artifacts/embeddings/speaker_cv/fold_1/metadata.json`
- Lip command:
  `python3 scripts/09_train_lip_fold_embeddings.py --config configs/real_embeddings.local.yaml --fold 1`

- `artifacts/embeddings/speaker_cv/fold_1/lip_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_1/mouth_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_1/uwb_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_1/mmwave_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_1/laser_embeddings.npz`

## Fold 2

- Train speakers: [15, 8, 13, 11, 7, 20, 12, 19, 3, 5, 18, 2, 14, 9]
- Val speakers: [16, 10]
- Test speakers: [4, 1, 17, 6]
- Metadata: `artifacts/embeddings/speaker_cv/fold_2/metadata.json`
- Lip command:
  `python3 scripts/09_train_lip_fold_embeddings.py --config configs/real_embeddings.local.yaml --fold 2`

- `artifacts/embeddings/speaker_cv/fold_2/lip_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_2/mouth_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_2/uwb_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_2/mmwave_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_2/laser_embeddings.npz`

## Fold 3

- Train speakers: [15, 8, 13, 11, 7, 20, 4, 1, 17, 6, 18, 2, 14, 9]
- Val speakers: [16, 10]
- Test speakers: [12, 19, 3, 5]
- Metadata: `artifacts/embeddings/speaker_cv/fold_3/metadata.json`
- Lip command:
  `python3 scripts/09_train_lip_fold_embeddings.py --config configs/real_embeddings.local.yaml --fold 3`

- `artifacts/embeddings/speaker_cv/fold_3/lip_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_3/mouth_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_3/uwb_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_3/mmwave_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_3/laser_embeddings.npz`

## Fold 4

- Train speakers: [15, 8, 13, 11, 7, 20, 4, 1, 17, 6, 12, 19, 3, 5]
- Val speakers: [16, 10]
- Test speakers: [18, 2, 14, 9]
- Metadata: `artifacts/embeddings/speaker_cv/fold_4/metadata.json`
- Lip command:
  `python3 scripts/09_train_lip_fold_embeddings.py --config configs/real_embeddings.local.yaml --fold 4`

- `artifacts/embeddings/speaker_cv/fold_4/lip_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_4/mouth_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_4/uwb_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_4/mmwave_embeddings.npz`
- `artifacts/embeddings/speaker_cv/fold_4/laser_embeddings.npz`
