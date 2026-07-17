# HuBERT Teacher Target Extraction

Real utterance-level audio teacher targets were extracted from the audited RVTALL
audio manifest using `facebook/hubert-base-ls960`.

## Result

| Metric | Value |
|---|---:|
| Audio/manifest pairs | 596 |
| Speakers | 20 |
| Classes | 30 |
| Teacher target shape | 596 x 768 |
| Non-finite values | 0 |
| Duplicate speaker/group pairs | 0 |
| Mean target L2 norm | 6.491 |
| Mean per-feature standard deviation | 0.0703 |
| Saved target size | 1.7 MB |

The selected files contain 33.3 minutes of audio. Full extraction on Apple MPS took
28.6 seconds and used approximately 3.0 GB peak memory after the model was cached.

## Reproduction

Install the optional teacher dependencies:

```bash
python3 -m pip install -e '.[audio-teachers]'
```

Build the machine-local audio manifest and extract targets:

```bash
python3 scripts/20_build_audio_manifest.py \
  --config configs/real_embeddings.local.yaml

python3 scripts/19_extract_ssl_teacher_targets.py \
  --config configs/real_embeddings.local.yaml \
  --manifest artifacts/manifest_with_audio.csv \
  --model-name facebook/hubert-base-ls960 \
  --device mps \
  --local-files-only \
  --output artifacts/teacher_targets/facebook_hubert-base-ls960_targets.npz
```

The target NPZ and model cache are generated artifacts and are intentionally not
tracked by Git. Audio is used only to construct training targets; the sensor-student
inference path remains microphone-free.

## End-to-End Smoke Check

A three-epoch fold-0 student run verified that the real 768-dimensional targets work
with the existing sensor-student trainer. Validation target MSE decreased from 1.9166
to 1.8099. The run reached 56.7% validation accuracy and 39.3% test accuracy, but these
short-run values are pipeline checks only and are not five-fold research results.

## Interpretation

These are mean-pooled final HuBERT hidden states for each aligned utterance. They are
valid real speech-representation targets, but they are not yet temporal articulatory
trajectories. The next experiment is speaker-disjoint student training across all five
folds, followed by comparison with the strict supervised fusion baseline.
