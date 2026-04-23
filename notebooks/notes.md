# Silent Speech Decoding Project — Notes

## Project Overview
Multimodal silent speech decoding using the **RVTALL dataset**, which contains:
- UWB radar (7.5 GHz CIR)
- mmWave radar (77 GHz FMCW)
- Visual (Kinect RGB video)
- Lip landmarks (dlib 68-point, Kinect)
- Audio
- Laser

20 participants, each speaking 5 vowels, 15 words, and 16 sentences.
Dataset paper: https://www.nature.com/articles/s41597-023-02793-w

---

## Architecture Plan: Late Fusion

Each modality gets its own encoder that produces a fixed-size embedding (128-dim).
Those embeddings are concatenated and passed into a shared fusion MLP for final classification.
We are NOT doing frame-level synchronization across modalities — different sensors have
wildly different temporal lengths (e.g. radar ~0.5s vs video ~5-7s for the same sample).
Instead, each modality encoder compresses its full sequence into one 128-dim vector.

```
Lip landmarks  → LSTM encoder   → 128-dim embedding ─┐
Radar (UWB)    → 1D CNN/LSTM    → 128-dim embedding  ├→ concat → fusion MLP → prediction
Radar (mmWave) → 2D CNN         → 128-dim embedding  │
Laser          → 1D CNN/LSTM    → 128-dim embedding ─┘
```

Currently only the lip landmark encoder has been built. Other modalities are next.

---

## Terminology Clarifications
- **Feature extraction**: manually computing descriptive numbers (height, curvature, etc.) — what the original script did
- **Embedding**: a learned fixed-size vector output by a neural network — what the LSTM produces
- **Encoding**: same thing as embedding, used interchangeably
- **Classification head**: the final Linear layer (128 → num_classes) used only during training to give the model a task to optimize. Discarded at inference — only the 128-dim embedding before it is used for fusion.
- **Late fusion**: each modality is processed independently by its own model; outputs are combined afterward. Opposite of early fusion (combining raw signals) or mid fusion (combining partway through).

---

## Dataset Directory Structure (Lip Landmarks)

```
src/data/RVTALL/Processed_cut_data/kinect_processed/
└── {user_id}/                        # 1 through 20
    └── {sentences1-10 | vowels1-5 | words1-15}/
        └── videos/
            └── video_* OR video_proc_*/   # see naming note below
                └── landmarkers_cv/
                    └── 0000.npy, 0001.npy, ...   # one .npy per frame
```

Each `.npy` file is shape `(68, 2)` — 68 dlib face landmarks, each with (x, y).
Lip region = indices 48:68 → 20 points → shape `(20, 2)`.

### IMPORTANT: Video folder naming differs by user
- **User 1**: folders are named `video_0`, `video_1`, etc. → glob pattern: `video_[0-9]*`
- **Users 2-20**: folders are named `video_proc_0`, `video_proc_1`, etc. → glob pattern: `video_proc_*`
- Varying numbers of video folders per utterance — handled automatically by glob (no fixed count assumed)

---

## Lip Landmark LSTM Encoder (`lip_landmark_lstm.py`)

### Input
- Each sample = one `landmarkers_cv` folder = one utterance by one user
- Raw input per frame: 20 landmark (x,y) pairs flattened → (40,) + velocity (40,) = **(T, 80)**
- Velocity = finite difference of coordinates frame-to-frame (not hand-crafted, just raw change)
- Landmarks are normalized per-frame: centered at centroid, scaled by max distance

### Architecture
```
Input (T, 80)
→ LayerNorm
→ BiLSTM (2 layers, hidden=256 per direction)
→ last hidden state: concat fwd + bwd → (512,)
→ Dropout
→ Linear(512 → 128) + LayerNorm
→ L2 normalize   ← THIS IS THE EMBEDDING (fusion input)
→ Linear(128 → num_classes)  ← classification head, training only
```

### Labels
- One unified label set across all group types: `sentences1`, `vowels3`, `words7` etc. → 36 classes
- Label map saved to `lip_label_map.json`

### Train/Val/Test Split
- Split is BY USER (not by sample) to test speaker-independent generalization
- Val: users 17-18 | Test: users 19-20 | Train: users 1-16
- During initial debugging, switched to random 75/15/10 split to confirm model can learn at all

### Data Augmentation (training only)
- Random temporal crop: keep 80-100% of frames
- Small Gaussian jitter on coordinates (std=0.005)

### Training Details
- Loss: CrossEntropyLoss with label smoothing 0.1
- Optimizer: AdamW, lr=3e-4, weight_decay=1e-4
- Scheduler: CosineAnnealingLR
- Gradient clipping: max_norm=5.0
- Early stopping: patience=20

### Outputs
- `lip_lstm_model.pt` — trained weights
- `lip_embeddings.npz` — embeddings for every sample
  - `data['embeddings']` → shape (N, 128) — this is the fusion input
  - `data['labels']`, `data['user_ids']`, `data['group_names']`, `data['video_names']`
- `lip_label_map.json` — class index → label string

---

## Issues Encountered & Fixed

### 1. Near-zero val accuracy (0.040), early stop at epoch 13
- **Cause**: glob pattern `video_*` was not matching `video_proc_*` folders for users 2-20,
  so almost all samples were silently skipped
- **Fix**: separate glob patterns for user 1 vs users 2-20 (see naming note above)

### 2. User-based split too harsh for initial debugging
- With only 16 training users and val on 2 users, val set was too small and unrepresentative
- **Fix**: use random split first to confirm model learns, then switch back to user-based

---

## Next Steps (as of initial build)
1. Confirm lip LSTM trains correctly with the glob fix (watch "Loaded X samples" count)
2. Verify val accuracy climbs meaningfully by epoch 20 with random split
3. Switch back to user-based split for final training
4. Build encoders for remaining modalities (radar, laser) following same pattern
5. Build fusion MLP that takes concatenated embeddings from all modalities
6. Potential novel angle: modality dropout robustness (model works when a modality is missing)

---

## Phase 2: All Encoders Built → First Fusion Attempt (`fusionMLP.py`)

All 5 modality encoders were completed and each independently produced a 128-dim embedding:
- **Radar (mmWave)**: 2D CNN over range-Doppler maps
- **Laser**: 1D CNN/LSTM over time-series signal
- **Mouth (video frames)**: CNN over cropped mouth region frames
- **Lip landmarks**: BiLSTM (described above)
- **UWB radar**: 1D CNN/LSTM over CIR time-series

These were concatenated (640-dim total) and fed into `fusionMLP.py` — a Transformer-based
fusion head with ~440K parameters.

**Result: ~40% val accuracy.**

### Why it failed
440K parameters trying to learn from ~3500 training samples across 16 speakers.
The model memorized which speakers said what (speaker identity leaks through embeddings)
rather than learning utterance-level features. Classic overfitting, compounded by the
speaker-disjoint eval: val/test speakers are completely unseen during training.

---

## Phase 3: Rethinking Fusion — `fusionGate.py`

### Core idea: prototypes instead of a learned classifier

For each class (e.g. "word3") and each modality, compute the **mean training embedding**
(the "prototype"). At test time, measure cosine similarity of a test embedding to each
class prototype — that's a "vote" from that modality. No learned parameters = no overfitting.

### Three no-training fusion baselines

- **Equal-weight**: average the 5 modalities' prototype votes equally
- **Borda count**: rank each modality's class predictions 1–30, sum the ranks across modalities.
  Resistant to one overconfident-but-wrong modality dominating the fusion.
- **Consistency-weighted**: weight each modality by how much it agrees with the other 4.
  If radar/mouth/laser all predict "word3" but lip predicts "word7", lip is automatically
  down-weighted for that sample.

### Trained gate
A tiny MLP (~43K params) that learns per-sample modality weights, trained with LOSO
(Leave-One-Speaker-Out) prototypes so it never sees the held-out speaker's embeddings
during training. In practice it barely beat the no-training methods — the val set is only
59 samples, which is too small to train a reliable gate.

### Results

| Method | Val | Test |
|--------|-----|------|
| fusionMLP (Transformer) | ~40% | — |
| Equal-weight | 61.0% | 76.7% |
| Borda count | 67.8% | 75.0% |
| **Consistency-weighted** | **66.1%** | **78.3%** |
| Trained gate | 59.3% | 73.3% |

**Best result: 78.3% test accuracy** (consistency-weighted, no learned parameters).

---

## Phase 4: Improving the Encoders — v2 Models

The fusion ceiling was limited by encoder quality. The original lip and UWB encoders
overfit to training speakers, performing poorly on unseen val/test speakers.

### `liplandmarkLSTM_v2.py` — three improvements

1. **Temporal attention pooling**: instead of taking only the final BiLSTM hidden state,
   learn a weighted average over all timesteps. Focuses on peak articulation frames
   rather than being dominated by the end of the utterance.

2. **Supervised Contrastive Loss (SupCon)**: pulls same-word embeddings from *different
   speakers* together in embedding space. Directly optimizes for speaker-invariant
   clustering — what the fusion layer needs.

3. **Domain Adversarial Training (DANN)**: a speaker-ID classifier is attached via a
   Gradient Reversal Layer. The encoder is penalized for encoding speaker identity,
   forcing it to produce speaker-agnostic embeddings.

**Result: lip encoder went from ~15% → 42.8% test accuracy** (speaker-disjoint).

### `uwbLSTMCNN_v2.py` — improvements + bug fix

- Added residual 2D CNN blocks (ResBlock2D) for better feature extraction
- Added temporal attention pooling and DANN head (same as lip v2)
- Improved preprocessing: per-antenna per-bin z-score normalization + ±3σ clipping
- **Fixed a critical bug**: `_time_out_lengths()` was computing CNN output lengths using
  range strides instead of time strides, causing the packed LSTM to crash with
  `RuntimeError: start + length exceeds dimension size`
- Training was killed early (checkpoint at ~60% of epochs) to save time, so UWB v2
  results are partial

### `fusionE2E.py` — joint fine-tuning (attempted)

The idea: re-train the lip encoder jointly with the fusion loss, so its embeddings are
optimized to complement the other 4 modalities rather than just lip classification.

**What happened**: the gate collapsed — it assigned ~90% weight to lip and ignored
everything else. Backprop only flows through the lip encoder (the others are frozen NPZs),
so the gate learns to just trust the one encoder that's actually updating.

The resulting `lip_embeddings_e2e.npz` is used by fusionGate.py but offers minimal
improvement over the v2 embeddings.

---

## Final Architecture (as presented)

```
Lip landmarks  → BiLSTM v2 (DANN + SupCon + Attn.) → 128-dim ─┐
Radar (mmWave) → 2D CNN encoder (frozen)            → 128-dim  │
Laser          → 1D CNN/LSTM encoder (frozen)       → 128-dim  ├→ Consistency-weighted
Mouth (video)  → CNN encoder (frozen)               → 128-dim  │   prototype fusion → prediction
UWB radar      → CNN/LSTM v2 (DANN + Attn.)         → 128-dim ─┘
```

Classification via per-class prototype cosine similarity, no learned fusion head.

**Final accuracy: 78.3% test, 66.1% val (30 classes, speaker-disjoint, chance = 3.3%)**