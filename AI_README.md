Here’s a crisp hand-off you can paste into any LLM to get them up to speed fast.

## Current interpretability status

- Strict encoder-disjoint supervised baseline: 63.9% validation-weighted fusion.
- Pooled HuBERT student: 64.0%; sparse causal-feature analysis complete.
- Four-segment temporal HuBERT student: 64.5%, with true-order segment cosine 0.346
  versus 0.086 for reversed order.
- Held-out sparse-feature exemplars, temporal probes, SAEs, and causal ablations are
  complete. Start with `reports/temporal_interpretability_batch.md`.
- Fold-specific temporal silent-sensor activations are now exposed and aligned to
  four-segment HuBERT targets. True-order cosine is 0.381 versus 0.047 reversed.
- Speaker-disjoint articulation probes are complete. Non-lip contactless sensors add
  `+0.076 R2` for lip motion beyond a class-and-position baseline, but do not reliably
  recover lip aperture or width. Start with `reports/temporal_sensor_interpretability.md`.
- The local release has no prompt text, TextGrids, or phoneme timestamps, so true forced
  alignment remains blocked on obtaining transcript/annotation metadata.
- Next decisive step: add multitask temporal/classification training to recover the
  classification drop (49.9% temporal-state student versus 64.5% fixed-embedding student),
  then run phoneme probes once prompt text or external alignments are available.

# Project snapshot

**Name:** silentSpeech
**Goal:** Goal: Learn and align multimodal signals for silent/speech interfaces (audio, lip landmarks/video, laser speckle, FMCW radar, UWB), with future EEG↔fMRI fusion. 

## Repo layout (top level)

```
.
├─ README.md
├─ requirements.txt
├─ notebooks/
│  └─ colab_starter.ipynb           # EDA + alignment + small evals (ready)
├─ scripts/
│  ├─ download_openneuro.sh
│  ├─ git_fix_parent_repo.sh
│  └─ prepare_rvtall.sh
└─ src/
   ├─ configs/
   │  ├─ .yaml
   │  └─ .yaml     # main training config (edit here)
   ├─ data/
   │  ├─ RVTALL/Processed_cut_data/  # main dataset root (see below)
   │  ├─ emg_throat.py
   │  ├─ inner_speech_eeg_fmri.py    # placeholder for future fusion
   │  ├─ rvtall.py
   │  └─ rvtall_auto.py
   ├─ models/
   │  ├─ encoders.py                 # per-modality encoders
   │  ├─ decoders.py
   │  └─ fusion.py                   # fusion / projection heads
   ├─ scripts/
   │  ├─ fetch_openneuro.py
   │  └─ validate_rvtall.py          # dataset presence check
   ├─ training/
   │  ├─ train_contrastive.py        # entrypoint for contrastive runs
   │  └─ train_asr.py
   └─ utils/
      ├─ paths.py
      └─ rvtall_io.py                # robust loaders & path resolution (custom)
```

# Dataset (RVTALL) – what’s present

**Root:** `src/data/RVTALL/Processed_cut_data/`

Subfolders (each with subject IDs `1/…/N` → corpus dirs):

* `kinect_processed/<set>/<corpus>/`

  * `audios/audio_proc_<i>.wav`
  * `landmarkers/land_proc_<i>.csv`
  * `videos/video_<i>/...` (boxes/frames/mouth PNGs)
* `laser_processed/<set>/<corpus>/sample_*.npy`
* `radar_processed/<set>/<corpus>/sample_*.npy` (FMCW)
* `uwb_processed/<set>/<corpus>/sample_*.npy`

Notes:

* Corpus names are inconsistent (e.g., `word5` vs `word_5`, `sentence` vs `sentences`).
* Our `utils/rvtall_io.py` resolves these variants and **pairs** audio/landmark files by shared index `<i>`.

# Preprocessing & alignment

* We **do not** re-run raw → processed pipelines; data already pre-cut by authors.
* We verified **time coherence** with a notebook EDA:

  * Lip aperture (upper–lower lip depth from CSV) vs. audio RMS
  * Optional overlays with laser/radar
  * Cross-correlation lags; histograms per semantic type (vowel/word/sentence)
* Results: “time-synched enough” for training; optional per-corpus mean offsets can be saved for future compensation.

# Notebook (EDA) – what it does

`notebooks/colab_starter.ipynb` provides:

* Robust pathing (adds `src/` to `sys.path`; auto-finds RVTALL base)
* Single-sample overlays (Audio vs Lip; optional Laser/Radar)
* Sampled aggregated stats:

  * Lip→Audio lag per pair; DataFrame with (set_id, corpus, semantic, pair_idx, lag_ms, modality presence)
  * Histograms/boxplots; alignment % within ±50/±100/±150 ms
  * Small cross-modal correlations (audio↔laser/radar) on a few samples
* Optional: save mean offsets for **sampled** corpora only (keeps runtime small)

# Utilities (critical)

`src/utils/rvtall_io.py`:

* `find_rvtall_base()` – locate dataset from anywhere in repo
* `resolve_corpus_dir()` – handles `word5` vs `word_5`, `sentence(s)`*
* `kinect_pair_index_paths()` – returns matched `(idx, audio, landmark)` per corpus
* `first_npy()` – fetch first `sample_*.npy` for laser/radar/uwb in the matching corpus
* `load_audio_env()`, `load_lip_aperture_csv()`, `zscore()` – convenience loaders

# Train

**Config:** `src/configs___.yaml` (adjust batch/epochs/precision)
**Run:**

```bash
python -m src.scripts.validate_rvtall --base src/data/RVTALL
python -m src.training.train_ --config src/configs/___.yaml
```

Recommended small settings (RTX 3050 Ti ASSUME MAC + google colab support):

* `batch_size: 8–12`, `num_workers: 0` (Windows), `max_epochs: 5` (sanity) → then ~25
* Mixed precision (`16-mixed`) if supported

# Eval plan (alpha gates + reporting)

**Core retrieval (test):**

* Pairwise R@1/R@5/MedR/MRR for audio↔lip, audio↔laser, audio↔radar
* Lip→Audio **lag**: % within ±50/±100 ms; Lag RMSE
* Cross-modal corr: mean Pearson r (audio envelope vs laser/radar envelopes)

**Downstream mini-tasks:**

* Vowel classification (probe on frozen embeddings): accuracy
* Word shortlist classification: Top-1/Top-5
* (Optional)

**Interpretability ():**


**Alpha acceptance (initial targets):**

* audio↔lip: R@5 ≥ 45% (R@1 ≥ 20%); lag within ±100 ms ≥ 80%
* Enrichment: ≥2 clusters with p < 1e-5; ARI ≥ 0.25; stability AMI ≥ 0.60
* Adding a modality (e.g., laser) improves any pair’s R@5 by ≥ +5% absolute **or** vowel acc by ≥ +3% absolute

# Scaling modalities

1. Start: `modalities: ["audio","kinect"]`
2. Add **laser** →
3. Add **radar**, then **UWB**
4. Keep eval deltas; only keep a modality if it helps metrics or interpretability

# EEG ↔ fMRI (future track)

* **EEG**: preprocess (re-ref, band/line, ICA),
* **fMRI**: fMRIPrep →
* **Losses**: 
* **Evals**: per-ROI r / R² lift over GLM; EEG↔fMRI R@K; cluster ARI/NMI vs canonical atlases

# Quick facts to relay

* Data counts (approx): `audio wavs ~5300`, `mouth pngs ~468k`, `radar npy ~4956`, `uwb npy ~10538`, `laser npy ~5091`.
* Time alignment: **checked**; acceptable; optional per-corpus offset file saved for sampled corpora.
* Naming inconsistencies handled by `rvtall_io.py` (don’t assume exact filenames).
* Notebook provides **lightweight**, **sampled** stats (not full sweeps) to stay fast.

# One-liner for a new LLM agent

> “This repo trains a contrastive multimodal model on RVTALL (audio, lip landmarks/video, laser, radar, UWB). Use `rvtall_io.py` to resolve messy corpus names and pair files. The notebook verifies time alignment and logs sampled lag/correlation stats. Train with `train_contrastive.py` (start audio+lip), then add modalities with ___. Evaluate with retrieval (R@K), lag % within thresholds, small probes, and ____ for interpretability and model selection.”
