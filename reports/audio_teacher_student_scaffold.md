# Audio Teacher/Student Scaffold

This report documents the first working scaffold for the audio-teacher phase of the
silent speech interpretability project.

## What Exists Now

- Teacher-target NPZ schema:
  - `targets`: fixed-dimensional teacher vectors.
  - `labels`: class IDs.
  - `user_ids`: speaker IDs.
  - `group_names`: utterance/sample group keys.
  - `target_name`: teacher identifier.
- Synthetic teacher target generator:
  - `scripts/17_prepare_teacher_targets.py`
- SSL audio teacher extractor:
  - `scripts/19_extract_ssl_teacher_targets.py`
- Silent-sensor student trainer:
  - `scripts/18_train_teacher_student.py`
- Student model:
  - `silent_speech_interpretability.models.students.ArticulatoryStudent`

The scaffold uses the same fold-specific, encoder-disjoint split discipline as the
strict CV baseline. By default it uses the current fusion modalities and excludes mouth:

- lip
- UWB
- mmWave
- laser

## Smoke Run

The local smoke run used deterministic synthetic teacher targets, not a real audio model:

```text
python3 scripts/17_prepare_teacher_targets.py \
  --config configs/real_embeddings.local.yaml \
  --target-dim 32 \
  --output artifacts/teacher_targets/synthetic_audio_teacher_targets.npz

python3 scripts/18_train_teacher_student.py \
  --config configs/real_embeddings.local.yaml \
  --fold 0 \
  --max-epochs 3 \
  --batch-size 128
```

Smoke-run outputs:

```text
num_train: 392
num_val: 30
num_test: 117
val_accuracy: 53.3%
test_accuracy: 43.6%
```

These numbers are only a pipeline sanity check because the teacher targets are synthetic
and class-structured. They should not be interpreted as audio-teacher performance.

## What This Unlocks

The project now has the plumbing needed to swap synthetic targets for real audio-teacher
targets from HuBERT, wav2vec2, SPARC, Sylber, or another speech representation model.
The student training/evaluation path does not need to change as long as the real teacher
extractor writes the same NPZ schema.

## Real SSL Teacher Extraction Status

The SSL extraction path is implemented for HuBERT/wav2vec-style models through
`transformers`:

```text
python3 scripts/19_extract_ssl_teacher_targets.py \
  --config configs/real_embeddings.local.yaml \
  --model-name facebook/hubert-base-ls960 \
  --local-files-only
```

Current local status:

- `librosa` and `soundfile` are installed.
- `transformers` is installed and declared in the `audio-teachers` optional dependency.
- The RVTALL audio has been discovered and aligned in an audited local manifest.
- See `reports/audio_manifest_audit.md` for coverage and repetition matching details.
- Real HuBERT targets were extracted for 596 unique speaker/group pairs.
- See `reports/hubert_teacher_extraction.md` for target validation and runtime.

Real utterance-level SSL teacher extraction is no longer blocked.

## Next Steps

1. Train HuBERT-target students across all 5 speaker-disjoint folds.
2. Compare student target MSE and class accuracy against the strict fusion baseline.
3. Decide whether the next teacher experiment should use pooled vectors or temporal
   sequences.
4. Add SPARC or Sylber targets if their optional dependencies are available.
