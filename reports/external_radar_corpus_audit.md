# External Radar Corpus Audit

## Pairing result

- Samples: **3,000**
- Subjects: **2**
- Sessions: **3**
- Command classes: **50**
- Repetitions per subject/session/class: **10**
- Random binary-file inspection (12 files): **59–96 frames; 128–128 steps; 9–9 spectra**
- Pairing key: exact shared filename stem across radar, WAV audio, and text label files

Every manifest row has an existing radar file, paired audio file, and non-empty text label.
The manifest is sorted deterministically by subject, session, class, and repetition.

## Role in this project

This corpus provides an independent radar/audio test bed for the audio-teacher and
silent-sensor-student method. It changes the laboratory, language, speakers, command
inventory, and radar hardware relative to RVTALL. The planned evaluation holds out
recording sessions and uses audio only while creating teacher targets; inference uses
radar features alone.

## Scope limitation

The corpus has only two subjects. A positive result is therefore evidence of external
cross-session replication, not broad population-level speaker generalization.

## Provenance

- [Wagner et al., “Silent speech command word recognition using stepped frequency
  continuous wave radar,” *Scientific Reports* (2022)](https://pubmed.ncbi.nlm.nih.gov/35273225/).
- [Official supplementary implementation](https://github.com/TUD-STKS/radar_based_command_word_recognition)
  and corpus published by TU Dresden / VocalTractLab.
