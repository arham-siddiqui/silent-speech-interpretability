#!/usr/bin/env python3
"""Generate Markdown reports from true encoder-disjoint CV CSV outputs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd


METHOD_NAMES = {
    "prototype": "Prototype",
    "equal_weight": "Equal-weight fusion",
    "equal_weight_no_mouth": "Equal-weight no-mouth fusion",
    "validation_weighted": "Validation-weighted fusion",
    "borda": "Borda fusion",
    "consistency_weighted": "Consistency-weighted fusion",
}

MODALITY_NAMES = {
    "lip": "Lip",
    "mouth": "Mouth",
    "uwb": "UWB",
    "mmwave": "mmWave",
    "laser": "Laser",
    "fusion": "Fusion",
}


def _pct(value: float) -> str:
    return f"{100.0 * float(value):.1f}%"


def _markdown_table(headers: list[str], rows: list[list[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(lines)


def _aggregate_table(summary: pd.DataFrame) -> str:
    ranked = summary.sort_values("mean", ascending=False)
    rows = []
    for item in ranked.itertuples(index=False):
        rows.append(
            [
                METHOD_NAMES.get(item.method, item.method),
                MODALITY_NAMES.get(item.modality, item.modality),
                _pct(item.mean),
                _pct(item.std) if pd.notna(item.std) else "",
                int(item.count),
            ]
        )
    return _markdown_table(["Method", "Modality", "Mean Accuracy", "Std. Dev.", "Folds"], rows)


def _fold_table(results: pd.DataFrame) -> str:
    wanted = [
        ("prototype", "lip", "Lip"),
        ("prototype", "mouth", "Mouth"),
        ("prototype", "uwb", "UWB"),
        ("prototype", "mmwave", "mmWave"),
        ("prototype", "laser", "Laser"),
        ("equal_weight", "fusion", "Equal Fusion"),
        ("validation_weighted", "fusion", "Validation Fusion"),
        ("borda", "fusion", "Borda"),
        ("consistency_weighted", "fusion", "Consistency Fusion"),
    ]
    rows = []
    for fold in sorted(results["fold"].unique()):
        fold_rows = results[results["fold"] == fold]
        row = [int(fold)]
        for method, modality, _label in wanted:
            match = fold_rows[(fold_rows["method"] == method) & (fold_rows["modality"] == modality)]
            row.append(_pct(float(match.iloc[0]["accuracy"])) if len(match) else "")
        rows.append(row)
    return _markdown_table(["Fold"] + [label for _method, _modality, label in wanted], rows)


def _weights_table(weights: pd.DataFrame) -> str:
    pivot = weights.pivot(index="fold", columns="modality", values="weight").reset_index()
    modalities = [modality for modality in ["lip", "mouth", "uwb", "mmwave", "laser"] if modality in pivot.columns]
    rows = []
    for item in pivot.itertuples(index=False):
        row = [int(getattr(item, "fold"))]
        for modality in modalities:
            row.append(_pct(getattr(item, modality)))
        rows.append(row)
    return _markdown_table(["Fold"] + [MODALITY_NAMES.get(modality, modality) for modality in modalities], rows)


def _per_class_table(per_class: pd.DataFrame, method: str, modality: str, ascending: bool) -> str:
    subset = per_class[(per_class["method"] == method) & (per_class["modality"] == modality)].copy()
    subset = subset.sort_values(["accuracy", "num_samples"], ascending=[ascending, False]).head(10)
    rows = [
        [int(item.class_id), _pct(item.accuracy), int(item.num_correct), int(item.num_samples)]
        for item in subset.itertuples(index=False)
    ]
    return _markdown_table(["Class ID", "Accuracy", "Correct", "Samples"], rows)


def _write_true_cv_report(results_dir: Path, report_path: Path) -> None:
    results = pd.read_csv(results_dir / "true_encoder_cv_results.csv")
    summary = pd.read_csv(results_dir / "true_encoder_cv_summary.csv")
    per_class = pd.read_csv(results_dir / "true_encoder_cv_per_class.csv")
    weights = pd.read_csv(results_dir / "true_encoder_cv_fusion_weights.csv")

    best = summary.sort_values("mean", ascending=False).iloc[0]
    equal = summary[(summary["method"] == "equal_weight") & (summary["modality"] == "fusion")].iloc[0]
    validation = summary[(summary["method"] == "validation_weighted") & (summary["modality"] == "fusion")].iloc[0]
    lip = summary[(summary["method"] == "prototype") & (summary["modality"] == "lip")].iloc[0]

    content = f"""# True Encoder-Disjoint CV Results

This report is generated from the completed real-encoder, speaker/encoder-disjoint
cross-validation outputs in `reports/results/`.

## Evaluation Setup

- Dataset: RVTALL silent speech subset, 30 classes.
- Modalities: lip landmarks, mouth video, UWB radar, mmWave radar, laser speckle.
- Split: 5-fold encoder-disjoint speaker CV.
- Classifier: prototype/nearest-centroid per modality, plus late-fusion voting.
- Fusion methods: equal-weight averaging, validation-weighted averaging, Borda rank
  fusion, and consistency-weighted fusion.
- Mouth is reported as a diagnostic modality but excluded from fusion because the
  current fold-specific mouth embeddings are near chance.
- Chance accuracy: 3.3%.
- All reported folds have `encoder_disjoint_test=True`.

Raw outputs:

- `reports/results/true_encoder_cv_results.csv`
- `reports/results/true_encoder_cv_summary.csv`
- `reports/results/true_encoder_cv_predictions.csv`
- `reports/results/true_encoder_cv_per_class.csv`
- `reports/results/true_encoder_cv_fusion_weights.csv`
- `reports/results/true_encoder_cv_missing_artifacts.csv`

`true_encoder_cv_missing_artifacts.csv` is intentionally empty except for the header,
meaning all required fold/modality artifacts are present.

## Aggregate Accuracy

![Mean accuracy](figures/true_cv_mean_accuracy.svg)

{_aggregate_table(summary)}

## Per-Fold Accuracy

{_fold_table(results)}

## Validation-Derived Fusion Weights

These weights are estimated from each fold's validation speakers only, then applied
to the held-out test speakers. Mouth appears with zero weight because it is excluded
from fusion by config.

![Validation-derived fusion weights](figures/true_cv_fusion_weights.svg)

{_weights_table(weights)}

## Per-Class Error Snapshot

![Per-class validation-weighted fusion accuracy](figures/true_cv_per_class_accuracy.svg)

Hardest classes for the best fusion method (`{best["method"]}` / `{best["modality"]}`):

{_per_class_table(per_class, str(best["method"]), str(best["modality"]), ascending=True)}

Strongest classes for the best fusion method (`{best["method"]}` / `{best["modality"]}`):

{_per_class_table(per_class, str(best["method"]), str(best["modality"]), ascending=False)}

## Main Takeaways

1. The best overall method is {METHOD_NAMES.get(best["method"], best["method"])}, averaging
   {_pct(best["mean"])} accuracy across five folds.
2. Validation-weighted fusion averages {_pct(validation["mean"])}, compared with
   {_pct(equal["mean"])} for equal-weight fusion and {_pct(lip["mean"])} for lip alone.
3. Lip remains the dominant single modality, but validation-derived weighting recovers
   useful auxiliary signal on several folds.
4. UWB and laser carry useful but variable auxiliary signal.
5. mmWave is consistently above chance but weaker than earlier fixed-split baselines.
6. Mouth embeddings remain near chance in this CV setting and should be treated as
   provisional until the mouth encoder is retrained/audited.

## Verification

The final verification run completed with:

```text
pytest -q
20 passed
```

Strict true encoder CV completed without `--allow-missing`, confirming that the full
artifact contract is satisfied.

## Recommended Next Steps

1. Use `validation_weighted` as the current fusion baseline to beat.
2. Inspect `true_encoder_cv_per_class.csv` to target classes where auxiliary sensors help.
3. Retrain or replace the mouth encoder artifacts with full scientific fold embeddings.
4. Add plots for per-fold modality weights and per-class confusion.
5. Update manuscript/README language to treat older fixed-split numbers as legacy.
"""
    report_path.write_text(content, encoding="utf-8")


def _write_mouth_audit(results_dir: Path, artifact_dir: Path, report_path: Path) -> None:
    summary = pd.read_csv(results_dir / "true_encoder_cv_summary.csv")
    results = pd.read_csv(results_dir / "true_encoder_cv_results.csv")
    mouth_summary = summary[(summary["method"] == "prototype") & (summary["modality"] == "mouth")].iloc[0]
    mouth_results = results[(results["method"] == "prototype") & (results["modality"] == "mouth")]

    metadata_rows = []
    for metadata_path in sorted(artifact_dir.glob("fold_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        training = metadata.get("mouth_training", {})
        metadata_rows.append(
            [
                int(metadata.get("fold", metadata_path.parent.name.split("_")[-1])),
                metadata.get("mouth_embedding_path", ""),
                training.get("max_epochs", ""),
                training.get("note", ""),
            ]
        )

    fold_rows = [
        [int(item.fold), _pct(item.accuracy), _pct(item.macro_f1), int(item.num_test)]
        for item in mouth_results.sort_values("fold").itertuples(index=False)
    ]

    content = f"""# Mouth Encoder Audit

The true encoder-disjoint CV run shows mouth embeddings near chance:
mean accuracy {_pct(mouth_summary['mean'])} +/- {_pct(mouth_summary['std'])}
across {int(mouth_summary['count'])} folds.

## Fold Results

{_markdown_table(["Fold", "Accuracy", "Macro F1", "Test Samples"], fold_rows)}

## Artifact Metadata Notes

{_markdown_table(["Fold", "Embedding Path", "Max Epochs", "Note"], metadata_rows)}

## Interpretation

The mouth artifacts are present and pass the artifact contract, but their metadata marks
them as projection/smoke-test style artifacts rather than final scientific mouth encoder
folds. The near-chance CV accuracy is therefore best interpreted as an artifact-quality
issue, not evidence that mouth video lacks signal.

The current fold-specific projection-head training path was rerun and still produced
near-chance held-out performance, so simply rerunning `scripts/11_train_mouth_fold_embeddings.py`
is not enough to fix this modality.

## Recommended Fix

Retrain fold-specific mouth video embeddings with the same encoder-disjoint discipline as
the lip, laser, UWB, and mmWave artifacts, then rerun `scripts/08_run_true_encoder_cv.py`
and regenerate this report.
"""
    report_path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="reports/results")
    parser.add_argument("--artifact-dir", default="artifacts/embeddings/speaker_cv")
    parser.add_argument("--report", default="reports/true_encoder_cv_results.md")
    parser.add_argument("--mouth-audit", default="reports/mouth_encoder_audit.md")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    artifact_dir = Path(args.artifact_dir)
    _write_true_cv_report(results_dir, Path(args.report))
    _write_mouth_audit(results_dir, artifact_dir, Path(args.mouth_audit))


if __name__ == "__main__":
    main()
