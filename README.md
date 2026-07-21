# Silent Speech Decoding — Project Overview

**Goal**: Decode silent speech (intended but unvocalized utterances) across 30 classes —
5 vowels, 15 words, 10 sentences — using five sensor modalities simultaneously.
No microphone. No vocalization. Speaker-independent generalization required.

---

## 1. Dataset — RVTALL

**Paper**: https://www.nature.com/articles/s41597-023-02793-w

20 participants each performed 5 vowels, 15 words, and 10 sentences (30 classes total),
recorded simultaneously across six sensor modalities:

| Modality | Sensor | Signal type |
|----------|--------|-------------|
| Lip landmarks | Kinect + dlib | 68 facial keypoints per frame |
| Mouth video | Kinect RGB | Cropped mouth region frames |
| UWB radar | 7.5 GHz CIR | Range-time matrix, 2 antennas |
| mmWave radar | 77 GHz FMCW | Range-Doppler map |
| Laser speckle | 1D photodiode | Raw time-series signal |
| Audio | Microphone | Not used (silent speech task) |

**Train/val/test split — by speaker, not by sample:**
- Train: users 1–16 (16 speakers)
- Val: users 17–18 (2 speakers, never seen during training)
- Test: users 19–20 (2 speakers, never seen during training)

This is called a **speaker-disjoint** evaluation. It is deliberately hard — the model
must generalize to entirely new people, not just new recordings of familiar voices.
Chance accuracy = 1/30 = **3.3%**.

---

## 2. Encoders

Each modality gets its own dedicated encoder that compresses a variable-length
sensor recording into a fixed **128-dimensional embedding vector**. All encoders
are trained independently with a classification head (128 → 30 classes) that is
discarded after training — only the 128-dim embedding is kept for fusion.

This is called **late fusion**: each sensor is processed fully before anything is combined.

---

### 2a. Lip Landmark Encoder — `liplandmarkLSTM.py` / `liplandmarkLSTM_v2.py`

**Input**: Each frame, extract the 20 lip landmarks (indices 48–68 of dlib's 68-point
face model), normalize to be speaker/scale invariant, concatenate with per-frame
velocity → **(T, 80)** sequence.

**v1 Architecture**:
```
Input (T, 80) → LayerNorm → BiLSTM (2 layers, 256 hidden)
→ last hidden state (512,) → Dropout → Linear(512→128) + LayerNorm → L2 normalize
```

**v2 Improvements** (`liplandmarkLSTM_v2.py`):

The v1 encoder overfit to training speakers — it learned speaker-specific lip shapes
rather than utterance-level articulation patterns. Three changes fixed this:

1. **Temporal attention pooling** — instead of using only the last hidden state, learn a
   weighted average over all LSTM timesteps. Focuses on the most phonetically
   informative frames rather than the tail of the sequence.

2. **Supervised Contrastive Loss (SupCon)** — in addition to cross-entropy, a contrastive
   loss pulls embeddings of the same utterance type together *across different speakers*.
   This directly optimizes for speaker-invariant clustering in embedding space.

3. **Domain Adversarial Training (DANN)** — a speaker-ID classifier is attached via a
   Gradient Reversal Layer. The encoder is penalized for encoding who is speaking,
   forcing it to be speaker-agnostic.

**Accuracy (speaker-disjoint test set)**:

| Version | Test Accuracy |
|---------|--------------|
| Lip v1 | 14.6% |
| Lip v2 (DANN + SupCon + Attn.) | **42.8%** |

---

### 2b. Mouth Video Encoder

**Input**: Cropped mouth-region PNG frames → ResNet18 backbone (pretrained ImageNet)
→ 512-dim → MLP head → 128-dim embedding.

**Test accuracy**: ~46.7% (speaker-disjoint)

---

### 2c. UWB Radar Encoder — `uwbLSTMCNN.py` / `uwbLSTMCNN_v2.py`

**Input**: Range-time matrix from 2 antennas, shape (T, 2, 205). Each antenna captures
how radio reflections change over time as articulators (tongue, jaw, lips) move.

**Architecture**: 2D CNN over the range-time map → max-pool over range → BiLSTM over
time → 128-dim embedding.

**v2** added residual CNN blocks, temporal attention, DANN, and improved preprocessing
(per-bin z-score normalization + ±3σ clipping). Training was killed early due to time
constraints, so v2 results are partial. The v1 checkpoint is used in the final system.

**Test accuracy**: ~21.3% (speaker-disjoint, v1)

---

### 2d. mmWave Radar Encoder

**Input**: Range-Doppler maps (2D frequency-domain representation) — captures jaw/tongue
motion as Doppler shift patterns.

**Architecture**: 2D CNN → global pool → 128-dim embedding.

**Test accuracy**: ~40.0% (speaker-disjoint)

---

### 2e. Laser Speckle Encoder

**Input**: 1D raw photodiode signal over time — captures vocal fold vibration micro-motion.

**Architecture**: 1D strided CNN → BiLSTM → 128-dim embedding.

**Test accuracy**: ~45.0% (speaker-disjoint)

---

## 3. Embeddings

After training, each encoder is run over the full dataset (all 20 speakers) in inference
mode to extract embeddings. These are saved as `.npz` files:

| File | Source | Shape |
|------|--------|-------|
| `lip_embeddings_v2.npz` | Lip v2 encoder | (5300, 128) |
| `mouth_frame_embeddings_trained_36class.npz` | Mouth encoder | (5421, 128) |
| `uwb_embeddings.npz` | UWB v1 encoder | (5255, 128) |
| `radar_embeddings.npz` | mmWave encoder | (4956, 128) |
| `laser_embeddings.npz` | Laser encoder | (5091, 128) |

Each NPZ contains: `embeddings` (N, 128), `labels` (N,), `user_ids` (N,), `group_names` (N,).

The fusion layer never sees raw sensor data — it works entirely from these pre-computed
128-dim vectors. This means fusion can be developed and evaluated without re-running
the expensive encoder training.

---

## 4. Fusion Approaches

The central challenge: how to combine 5 independent 128-dim votes into a single
class prediction. We tried two fundamentally different approaches.

---

### 4a. Transformer Fusion — `fusionMLP.py` (abandoned)

Concatenate all 5 embeddings → 640-dim → Transformer encoder (~440K parameters)
→ classification head.

**Why it failed**: 440K parameters memorizing 3500 training samples across only 16 speakers.
The model learned *who is speaking* rather than *what they said*, because speaker identity
leaks through the embeddings. On unseen val speakers: **~40% accuracy**.

---

### 4b. Prototype-based Fusion — `fusionGate.py` (final system)

**Core idea**: for each class and each modality, compute the mean training embedding
(the *prototype*). At test time, measure cosine similarity of a test embedding to all
30 class prototypes — this gives a probability distribution over classes from that modality
(its "vote"). No learned parameters in the classification step = no overfitting.

Three ways to combine the 5 votes:

**Equal-weight** — average the 5 probability vectors directly. Every modality counts equally.

**Borda count** — convert each modality's probability vector to a ranking (1st–30th place),
sum ranks across modalities. Class with lowest total rank wins. Resistant to one
overconfident-but-wrong modality dominating.

**Consistency-weighted** — for each test sample, weight each modality by how much it
agrees with the other four. If 4 modalities predict "word3" but lip predicts "word7",
lip is automatically down-weighted for that sample. Weights computed on-the-fly per
sample — no training required.

**Trained gate** — a small MLP (~43K params) that learns modality weights from training
data using LOSO (Leave-One-Speaker-Out) prototypes. In practice barely beat the
no-training methods due to the tiny 59-sample validation set.

---

### 4c. End-to-End Joint Fine-tuning — `fusionE2E.py` (attempted)

Re-train the lip encoder jointly with the fusion loss, so its embeddings optimize for
complementing the other modalities rather than just lip classification alone.

**What happened**: the gate collapsed to ~90% weight on lip, ignoring all other modalities.
Since only the lip encoder receives gradients (the others are frozen NPZ files), the gate
learned to trust the only encoder that was actually updating. Minimal improvement over v2.

---

## 5. Accuracy Results

### Current strict encoder-disjoint CV result

The most reliable current evaluation is the completed 5-fold **true encoder-disjoint**
speaker CV. In this setting, each fold uses fold-specific encoder artifacts whose
encoder-training speakers do not overlap the held-out test speakers.

Full report: [`reports/true_encoder_cv_results.md`](reports/true_encoder_cv_results.md)
Error analysis: [`reports/true_cv_error_analysis.md`](reports/true_cv_error_analysis.md)

| Method | Modality | Mean Accuracy |
|--------|----------|--------------:|
| Validation-weighted fusion | Fusion | **63.9%** |
| Equal-weight fusion | Fusion | 61.9% |
| Prototype | Lip | 60.9% |
| Consistency-weighted fusion | Fusion | 57.5% |
| Borda fusion | Fusion | 47.0% |
| Prototype | UWB | 26.7% |
| Prototype | Laser | 24.3% |
| Prototype | mmWave | 15.7% |
| Prototype | Mouth | 5.0% |

The mouth result is provisional and mouth is excluded from fusion by default. The
current mouth fold artifacts are documented as projection/smoke-test artifacts,
not final scientific mouth encoder folds; see
[`reports/mouth_encoder_audit.md`](reports/mouth_encoder_audit.md).

### Legacy fixed-split result

The numbers below are older fixed-split results from the original project setup.
They are useful historical context, but they are less reliable than the true
encoder-disjoint CV above because the final test set was much smaller.

**Evaluation**: speaker-disjoint, 30 classes, chance = 3.3%.
Data: 3509 train / 59 val / 60 test samples (intersection of all 5 modalities).

### Individual modality accuracy (nearest-centroid, test set)

| Modality | Test Accuracy |
|----------|--------------|
| Lip v2 (DANN + SupCon + Attn.) | 42.8% |
| Mouth | 46.7% |
| Laser | 45.0% |
| mmWave radar | 40.0% |
| UWB radar | 21.3% |

### Fusion accuracy — progression

| Method | Val (v1 lip) | Test (v1 lip) | Val (v2 lip) | Test (v2 lip) |
|--------|-------------|--------------|-------------|--------------|
| Transformer (fusionMLP) | ~40% | — | — | — |
| Equal-weight | 55.9% | 58.3% | 61.0% | 76.7% |
| Borda count | 52.5% | 61.7% | 67.8% | 75.0% |
| **Consistency-weighted** | 57.6% | 58.3% | **66.1%** | **78.3%** |
| Trained gate | 54.2% | 58.3% | 59.3% | 73.3% |

**Best result: 78.3% test accuracy** — consistency-weighted prototype fusion with v2 lip encoder.

The +20 point jump from v1→v2 lip demonstrates the key insight: in late-fusion systems,
**encoder quality is the binding constraint**. The fusion method matters far less than
how well the individual encoders generalize across speakers.

---

## 6. Future Directions

**Audio teacher/student distillation** — the repo now includes a first scaffold for
teacher-target storage and silent-sensor student training. See
[`reports/audio_teacher_student_scaffold.md`](reports/audio_teacher_student_scaffold.md).
Real HuBERT targets have now been extracted for 596 aligned speaker/utterance pairs;
see [`reports/hubert_teacher_extraction.md`](reports/hubert_teacher_extraction.md).
Five-fold speaker-disjoint student training reached 64.0% mean accuracy versus 63.9%
for strict validation-weighted fusion; see
[`reports/hubert_teacher_student_cv.md`](reports/hubert_teacher_student_cv.md).
The first probe and modality-attribution batch is summarized in
[`reports/hubert_interpretability_summary.md`](reports/hubert_interpretability_summary.md).
Sparse bottleneck features and controlled causal ablations are reported in
[`reports/hubert_bottleneck_feature_causality.md`](reports/hubert_bottleneck_feature_causality.md).
Held-out feature exemplars and the four-segment temporal HuBERT experiment are summarized
in [`reports/temporal_interpretability_batch.md`](reports/temporal_interpretability_batch.md),
with temporal sparse-feature controls in
[`reports/hubert_temporal_feature_causality.md`](reports/hubert_temporal_feature_causality.md).
Fold-specific temporal states from lip, laser, mmWave, and UWB are now evaluated in
[`reports/temporal_sensor_interpretability.md`](reports/temporal_sensor_interpretability.md).
The temporal-state student improves true-order HuBERT cosine from 0.346 to 0.381, while
speaker-disjoint probes show that non-lip contactless sensors add `+0.076 R2` for lip
motion beyond a class-and-position baseline. The validation-selected multitask model
recovers temporal-state accuracy from 49.9% to 60.1% while slightly improving true-order
cosine from 0.381 to 0.386; see
[`reports/temporal_sensor_multitask.md`](reports/temporal_sensor_multitask.md).
A modality-specific attention follow-up reaches only 56.8% and 0.378 cosine, so the
multitask model remains selected. Its diffuse held-out weights are documented in
[`reports/temporal_sensor_attention_audit.md`](reports/temporal_sensor_attention_audit.md).
The published RVTALL prompts have now been recovered, reconciled with a cohort permutation
in the local processed audio folders, and CTC-aligned. Speaker-disjoint phonetic occupancy
probes find the strongest residual signal in the combined contactless sensors: macro
`R2 = 0.511` versus `0.500` for class+position alone, with the largest gains for stops
(`+0.037`), liquids (`+0.025`), silence (`+0.022`), and fricatives (`+0.019`). See
[`reports/temporal_phonetic_probes.md`](reports/temporal_phonetic_probes.md).
An alternate Wav2Vec2 teacher underperforms matched HuBERT transfer (47.9% versus 49.9%
accuracy; 0.290 versus 0.381 temporal cosine), so HuBERT remains selected; see
[`reports/audio_teacher_comparison.md`](reports/audio_teacher_comparison.md).

Reproduce the tracked alignment and probe batch with:

```bash
pip install -e '.[audio-teachers,interpretability,alignment]'
make audio-phonetic-batch
make wav2vec2-teacher-comparison
```

The Make targets use `--local-files-only`; cache `facebook/wav2vec2-base-960h` before
running them. Large generated targets and checkpoints stay ignored, while aggregate CSVs
under `reports/tables/` are tracked.

**Retrain UWB encoder fully** — the v2 UWB training was killed early. A fully converged
UWB v2 with DANN + attention would likely lift both individual UWB accuracy and fusion
accuracy further.

**Apply DANN + SupCon to all encoders** — we only improved the lip encoder. Applying the
same speaker-invariance techniques to mouth, radar, and laser should raise all individual
modality ceilings and therefore the fusion ceiling.

**Larger speaker cohort** — the trained gate underperformed because 59 val samples is
not enough to learn reliable modality weights. More speakers would enable proper
gating to outperform the no-training baselines.

**Stronger phoneme alignment** — broad temporal phonetic probes are now complete using
CTC word boundaries and uniformly interpolated ARPAbet phones. Exact acoustic phone
boundaries still require a phone-level aligner or manually checked TextGrids. That sharper
annotation is the next requirement before naming individual latent features as phonemes.

**Modality-specific temporal attention** — the first attention branch underperforms the
simpler multitask student and produces diffuse sensor/time weights. Keep it as a controlled
negative result; revisit gating only with a sharper synchronization or regularization
hypothesis.

**Temporal alignment** — modalities currently operate independently with no cross-modal
synchronization. Learned alignment (e.g. cross-attention between modality streams) could
allow the fusion layer to attend to the right time window in each sensor.

**Extend the true-CV pipeline** — the project now has 5-fold true encoder-disjoint
speaker CV. The next step is to add reliability plots, per-class confusion figures,
and full mouth encoder fold retraining so the strict CV report becomes the primary
scientific result.
