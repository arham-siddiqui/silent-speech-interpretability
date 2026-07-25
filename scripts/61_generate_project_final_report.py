#!/usr/bin/env python3
"""Generate the consolidated project-level findings report."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _metric(summary: pd.DataFrame, protocol: str, metric: str) -> pd.Series:
    rows = summary[(summary["protocol"] == protocol) & (summary["metric"] == metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {protocol}/{metric} summary row, found {len(rows)}")
    return rows.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--external-summary",
        default="reports/tables/external_radar_generalization_summary.csv",
    )
    parser.add_argument(
        "--output",
        default="reports/project_final_summary.md",
    )
    args = parser.parse_args()

    summary = pd.read_csv(args.external_summary)
    session_accuracy = _metric(summary, "session", "accuracy")
    session_cosine = _metric(summary, "session", "segment_cosine")
    session_residual = _metric(summary, "session", "residual_order_margin_reversed")
    subject_accuracy = _metric(summary, "subject", "accuracy")
    subject_order = _metric(summary, "subject", "order_margin_reversed")

    report = f"""# Silent Speech Interpretability: Consolidated Findings

## Project objective

Build an interpretable silent-speech system from RVTALL non-audio sensors, establish a
strict speaker/encoder-disjoint supervised baseline, and use audio teachers to determine
which speech and articulatory information survives in silent-sensor representations.

## Completed evidence

| Track | Evaluation | Main result | Status |
|---|---|---:|---|
| Supervised baseline | Five-fold true encoder-disjoint RVTALL CV | 63.9% validation-weighted fusion | Complete |
| Audio teacher/student | Speaker-disjoint HuBERT student | 64.0% pooled student accuracy | Complete |
| Ordered representation | Four-segment silent-sensor to HuBERT transfer | 0.381 true-order cosine | Complete |
| Multitask temporal model | Speaker-disjoint class and HuBERT prediction | 60.1% accuracy; 0.386 cosine | Complete |
| Phonetic interpretation | Strict fold-valid phone occupancy probes | +0.038 all-sensor residual R2 | Complete |
| Manual timing audit | 40 balanced recordings | Exact timing trails matched uniform by -0.0025 R2 | Complete |
| External session replication | Three seeds, held-out radar sessions | {100 * session_accuracy['mean']:.1f}% accuracy; {session_cosine['mean']:.3f} cosine | Complete |
| External speaker transfer | Three seeds, leave-one-subject-out radar | {100 * subject_accuracy['mean']:.1f}% accuracy; {subject_order['mean']:+.3f} order margin | Negative result |

## Supported conclusions

1. **The supervised baseline is reliable.** Validation-weighted late fusion reaches
   63.9% under true speaker- and encoder-disjoint CV. Mouth remains excluded from fusion
   because fold-specific artifacts stay near chance.
2. **Silent sensors predict useful speech-teacher structure.** HuBERT-aligned students
   retain class and coarse ordered information under speaker-disjoint RVTALL evaluation.
3. **The phonetic signal is coarse rather than frame-exact.** Contactless modalities add
   residual phonetic occupancy information, but exact phone boundaries do not outperform
   matched uniform timing. The manual audit confirms this boundary.
4. **The method replicates across recording sessions in an independent radar corpus.**
   Multi-seed accuracy is **{100 * session_accuracy['mean']:.1f}%**
   [{100 * session_accuracy['unit_bootstrap_95_low']:.1f}%,
   {100 * session_accuracy['unit_bootstrap_95_high']:.1f}%] versus 2% chance. The
   within-command residual order margin remains **{session_residual['mean']:+.3f}**.
5. **The current external radar representation is not speaker-independent.**
   Leave-one-subject-out accuracy is only **{100 * subject_accuracy['mean']:.1f}%**
   [{100 * subject_accuracy['unit_bootstrap_95_low']:.1f}%,
   {100 * subject_accuracy['unit_bootstrap_95_high']:.1f}%], and its temporal order
   interval includes zero.

## Claims not supported

- Broad external speaker generalization from radar.
- Exact phone tracking from the current four-segment representations.
- One-feature/one-phoneme selectivity.
- A conclusion that adding architectural attention alone improves interpretability.

## Technical interpretation

The project succeeds at strict RVTALL decoding and at identifying coarse, ordered
speech-related structure in silent sensors. External ablations show that S32 and
magnitude features carry most session-transfer performance, while S12 and temporal
differences are weaker alone. The collapse under unseen-subject transfer indicates that
speaker-specific radar response and geometry remain dominant nuisance variables.

## Highest-value continuation

1. Add speaker-invariant radar normalization and source-subject domain adversarial
   training, selected without target-subject validation.
2. Evaluate on a larger external speaker cohort; two subjects cannot support a
   population-level generalization claim.
3. Preserve the current interpretability boundary and avoid further exact-phone or
   neuron-selectivity claims without higher-resolution synchronized targets.

## Evidence map

- [`true_encoder_cv_results.md`](true_encoder_cv_results.md)
- [`temporal_sensor_interpretability.md`](temporal_sensor_interpretability.md)
- [`phone_ctc_interpretability.md`](phone_ctc_interpretability.md)
- [`phone_boundary_audit_results.md`](phone_boundary_audit_results.md)
- [`external_radar_hubert_replication.md`](external_radar_hubert_replication.md)
- [`external_radar_generalization.md`](external_radar_generalization.md)
"""
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(f"Saved consolidated project report to {output}")


if __name__ == "__main__":
    main()
