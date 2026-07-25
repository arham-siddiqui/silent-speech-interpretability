#!/usr/bin/env python3
"""Generate the polished, professor-facing final project report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _row(summary: pd.DataFrame, protocol: str, metric: str) -> pd.Series:
    rows = summary[(summary["protocol"] == protocol) & (summary["metric"] == metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {protocol}/{metric} row, found {len(rows)}")
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-summary",
        default="reports/tables/external_radar_generalization_summary.csv",
    )
    parser.add_argument(
        "--external-ablations",
        default="reports/tables/external_radar_feature_ablation.csv",
    )
    parser.add_argument("--output", default="reports/final_project_report.md")
    args = parser.parse_args()

    external = pd.read_csv(args.external_summary)
    ablations = pd.read_csv(args.external_ablations)
    session_accuracy = _row(external, "session", "accuracy")
    session_cosine = _row(external, "session", "segment_cosine")
    session_order = _row(external, "session", "order_margin_reversed")
    session_residual = _row(external, "session", "residual_order_margin_reversed")
    subject_accuracy = _row(external, "subject", "accuracy")
    subject_cosine = _row(external, "subject", "segment_cosine")
    subject_order = _row(external, "subject", "order_margin_reversed")
    ablation_summary = (
        ablations.groupby("feature_set")
        .agg(accuracy=("accuracy", "mean"), cosine=("segment_cosine", "mean"))
        .sort_values("accuracy", ascending=False)
    )

    report = f"""# Silent Speech Interpretability

## Final Research Report

### Executive summary

This project converts an existing multimodal classifier into a reproducible,
interpretable framework for **contactless / microphone-free speech decoding**. It
establishes a strict speaker- and encoder-disjoint supervised baseline, trains
non-audio sensor students against audio-derived speech representations, tests what
phonetic and articulatory information those students retain, and replicates the method
on an independent radar corpus.

The strongest RVTALL baseline reaches **63.9%** accuracy across 30 classes. HuBERT
teacher/student experiments show that silent sensors retain class information and
coarse ordered speech structure. Phone-level and manual timing controls reject a
stronger exact-tracking interpretation. External radar experiments replicate the method
across recording sessions at **{100 * session_accuracy['mean']:.1f}%** accuracy, but
leave-one-subject-out performance falls to **{100 * subject_accuracy['mean']:.1f}%**,
near the 2% chance level. The principal unresolved limitation is therefore
speaker-specific sensor geometry, not a lack of within-speaker session robustness.

## 1. Research questions

1. Can non-audio RVTALL sensors decode 30 speech classes under genuinely
   speaker-disjoint and encoder-disjoint evaluation?
2. Can audio teachers transfer useful speech representations into sensor-only students?
3. Which articulatory, temporal, and phonetic properties are recoverable from the learned
   sensor representations?
4. Do the resulting conclusions replicate under a different laboratory, language,
   command inventory, radar system, and speaker cohort?

Audio is used only to create teacher targets during training and evaluation. Every
student inference path consumes non-audio sensor features only.

## 2. Data and evaluation design

### RVTALL

The internal experiments use the 20-participant
[RVTALL corpus](https://www.nature.com/articles/s41597-023-02793-w), with 30 classes:
5 vowels, 15 words, and 10 sentences. Available non-audio modalities are lip
landmarks, mouth video, UWB radar, mmWave/FMCW radar, and laser speckle.

The primary supervised evaluation is five-fold true encoder-disjoint CV. Test speakers
are absent not only from fusion fitting but also from the modality encoders that produce
their fold artifacts. This closes the most important leakage path in the original
fixed-embedding evaluation.

### External radar corpus

External validation uses the
[Wagner et al. radar command-word corpus](https://pubmed.ncbi.nlm.nih.gov/35273225/):
3,000 paired radar/audio samples from 2 subjects, 3 sessions, 50 German commands, and
10 repetitions per subject/session/class.

Two protocols answer different questions:

- **Session-held-out:** familiar subjects, unseen recording session.
- **Subject-held-out:** two source-subject sessions train, the third source session
  validates, and every session from the other subject tests.

All external estimates use three optimization seeds. Empirical intervals resample held-out
sessions or subjects after averaging seeds; because there are only three sessions and two
subjects, they are corpus-level diagnostics rather than population confidence intervals.

## 3. System

### Supervised baseline

Each modality produces a fixed-dimensional representation. Prototype classifiers compare
held-out embeddings with train-only class centroids using cosine similarity. Equal,
Borda, consistency-weighted, and validation-weighted late fusion combine modality votes.
Mouth is reported diagnostically but excluded from the selected fusion because
fold-specific mouth artifacts remain near chance.

### Audio teacher and sensor students

HuBERT is the selected audio teacher; a matched Wav2Vec2 comparison underperforms it.
Students map silent-sensor states into pooled or four-segment HuBERT targets while also
predicting utterance class. Four relative-time segments test ordered information without
claiming frame-exact synchronization.

### Interpretability

The analysis stack includes:

- speaker-disjoint linear probes;
- modality attribution and ablation;
- sparse bottleneck autoencoders;
- feature ranking and causal ablation;
- temporal articulation and phonetic occupancy probes;
- CTC-derived phone targets and matched uniform-timing controls;
- a balanced manual audit of 40 phone-boundary recordings.

### External radar representation

The external student uses the source implementation's S12 and S32 radargrams. Complex
magnitudes are log-scaled and standardized over time; magnitude and first-difference
features are pooled into 16 frequency bands and four relative-time segments.

## 4. Results

| Question | Evaluation | Result | Conclusion |
|---|---|---:|---|
| Strict decoding | True encoder-disjoint RVTALL CV | **63.9%** validation-weighted fusion | Supported |
| Pooled teacher transfer | Speaker-disjoint HuBERT student | **64.0%** accuracy | Supported |
| Ordered teacher transfer | Temporal sensor states | **0.381** true-order cosine | Supported |
| Multitask temporal model | Class + HuBERT targets | **60.1%**, **0.386** cosine | Supported |
| Fold-valid phonetic occupancy | All modalities beyond class+position | **+0.038 R2** | Coarse signal |
| Manual exact-timing audit | Exact versus matched uniform | **-0.0025 R2** | Exact timing rejected |
| External session transfer | Three seeds, held-out sessions | **{100 * session_accuracy['mean']:.1f}%** accuracy | Supported |
| External speaker transfer | Three seeds, held-out subjects | **{100 * subject_accuracy['mean']:.1f}%** accuracy | Not supported |

### 4.1 Strict supervised baseline

Validation-weighted fusion is the selected baseline at 63.9%, followed by no-mouth
equal-weight fusion at 61.9% and lip prototypes at 60.9%. The result is now grounded in
fold-specific encoder artifacts rather than a single small fixed test set.

### 4.2 Teacher/student interpretation

HuBERT-aligned students preserve useful class and coarse temporal structure. The
temporal-state student reaches 0.381 true-order cosine versus 0.047 reversed in the
internal experiment. Multitask training recovers classification performance without
sacrificing alignment. Modality-specific attention does not improve the simpler
multitask model and remains a controlled negative result.

### 4.3 Phonetic scope

Strict fold-valid probes find residual broad-phone occupancy information beyond
class and relative position. However, exact CTC phone timing does not outperform a
same-pairs uniform timing control. Manual review accepted 29 recordings, corrected 5,
and excluded 6; correction changes residual R2 by effectively zero. The supported
interpretation is **coarse ordered phonetic occupancy**, not exact phone tracking or
one-feature/one-phoneme selectivity.

### 4.4 External generalization

![External radar generalization](figures/final_external_generalization.svg)

Session-held-out radar accuracy is **{100 * session_accuracy['mean']:.1f}%**
[{100 * session_accuracy['unit_bootstrap_95_low']:.1f}%,
{100 * session_accuracy['unit_bootstrap_95_high']:.1f}%] versus 2% chance.
True-order HuBERT cosine is **{session_cosine['mean']:.3f}**, and the
true-minus-reversed margin is **{session_order['mean']:+.3f}**. After removing
train-only command prototypes, the residual order margin remains
**{session_residual['mean']:+.3f}**.

In contrast, subject-held-out accuracy is **{100 * subject_accuracy['mean']:.1f}%**
[{100 * subject_accuracy['unit_bootstrap_95_low']:.1f}%,
{100 * subject_accuracy['unit_bootstrap_95_high']:.1f}%]. HuBERT cosine is
**{subject_cosine['mean']:.3f}**, and order margin is **{subject_order['mean']:+.3f}**,
whose empirical interval crosses zero. The external result supports session robustness,
not unseen-speaker generalization.

### 4.5 External feature ablations

![External radar feature ablations](figures/final_external_feature_ablation.svg)

S32 alone retains **{100 * ablation_summary.loc['s32', 'accuracy']:.1f}%** accuracy,
essentially matching the combined input for seed 42. Magnitude-only features reach
**{100 * ablation_summary.loc['magnitude', 'accuracy']:.1f}%**, while delta-only and
S12-only features fall to **{100 * ablation_summary.loc['delta', 'accuracy']:.1f}%**
and **{100 * ablation_summary.loc['s12', 'accuracy']:.1f}%**. S32 magnitude structure
therefore carries most standalone session-transfer information.

## 5. Supported and unsupported claims

### Supported

- The selected non-audio fusion baseline generalizes across unseen RVTALL speakers under
  encoder-disjoint evaluation.
- Silent-sensor students recover class information and coarse ordered HuBERT structure.
- Contactless modalities add modest broad-phone occupancy information beyond lexical
  class and relative position.
- The radar teacher/student method replicates across recording sessions in an
  independent corpus.

### Not supported

- Broad external radar generalization to unseen speakers.
- Exact phone tracking from four-segment sensor representations.
- Individual sparse features as stable phoneme-selective units.
- A claim that attention or additional architectural complexity automatically improves
  interpretability.

## 6. Reproducibility

Install the project and optional research dependencies:

```bash
python3 -m pip install -e '.[audio-teachers,interpretability,alignment]'
```

Run lightweight verification:

```bash
make test
```

Reproduce the primary internal and external report layers:

```bash
make true-cv-artifacts
make phone-boundary-analysis
make external-radar-replication
make external-radar-validation-batch
make final-package
```

Large corpora, embeddings, teacher targets, activations, and checkpoints remain ignored.
Tracked Markdown, CSV, SVG, and notebook outputs contain the compact evidence needed for
review. Model downloads and long training runs should be runtime-piloted before launch.

## 7. Limitations and continuation

The internal cohort contains 20 speakers, while the external corpus contains only two.
External intervals cannot support population inference. The highest-value technical
continuation is source-only speaker-invariant radar normalization, optionally combined
with domain-adversarial training, followed by evaluation on a larger external cohort.
Target-speaker data must remain absent from model selection.

SPARC and Sylber remain optional teacher paths rather than required dependencies. HuBERT
provides the complete real-audio teacher result, while synthetic fixtures preserve local
testability when optional models or real data are absent.

## 8. Final conclusion

The project meets its central research objective: it provides a reproducible,
speaker-disjoint contactless speech baseline and a working interpretability framework
that connects non-audio sensors to audio-derived speech structure. The evidence supports
coarse ordered articulatory/phonetic information, with clear negative controls against
exact timing and speaker-independent external radar transfer. That boundary is the main
scientific result, not a defect to hide.
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Saved final project report to {output}")


if __name__ == "__main__":
    main()
