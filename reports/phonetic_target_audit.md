# Phonetic Target Audit

- Usable paired recordings: **596**.
- Relative temporal segments: **4**.
- Broad phonetic features: **8** (vowel, stop, fricative, affricate, nasal, liquid, glide, silence).
- CTC word-aligned recordings: **497**.
- Known isolated-vowel recordings: **99**.
- Recordings meeting the main confidence cutoff (>=0.05): **528**.

Word boundaries are constrained Viterbi alignments from `facebook/wav2vec2-base-960h`.
ARPAbet phones are distributed uniformly inside each aligned word interval. These are
therefore **interpolated phone occupancy targets**, not acoustically resolved phone
boundaries. The main probes retain only targets with alignment confidence >=0.05; isolated
vowels use the known vowel identity over the silence-trimmed acoustic interval.
