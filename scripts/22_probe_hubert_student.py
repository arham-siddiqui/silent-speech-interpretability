#!/usr/bin/env python3
"""Extract HuBERT-student activations and run frozen linear probes."""

from __future__ import annotations

import argparse
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.embeddings import load_embedding_repetitions, mean_eval_arrays
from silent_speech_interpretability.data.manifest import build_manifest, resolve_embedding_paths
from silent_speech_interpretability.data.splits import make_speaker_kfold_splits
from silent_speech_interpretability.evals.true_cv import configured_fold_embedding_paths
from silent_speech_interpretability.interp.activations import extract_student_activations
from silent_speech_interpretability.interp.probes import content_heldout_indices, fit_linear_probe
from silent_speech_interpretability.models.students.articulatory_student import ArticulatoryStudent
from silent_speech_interpretability.models.teachers.teacher_targets import common_teacher_pairs, load_teacher_targets, teacher_arrays


LAYER_ORDER = ["sensor_input", "hidden", "bottleneck", "predicted_hubert", "teacher_hubert"]


def _type_label(group_name: str) -> int:
    name = group_name.lower()
    if name.startswith("vowel"):
        return 0
    if name.startswith("word"):
        return 1
    if name.startswith("sentence"):
        return 2
    raise ValueError(f"Unknown utterance type for group {group_name!r}")


def _split_arrays(payloads, teacher, modalities, pairs):
    modality_arrays = []
    for modality in modalities:
        values, _labels = mean_eval_arrays(payloads[modality], pairs)
        modality_arrays.append(values)
    targets, labels = teacher_arrays(teacher, pairs)
    return {
        "inputs": np.concatenate(modality_arrays, axis=1).astype(np.float32),
        "teacher_hubert": targets,
        "labels": labels,
        "type_labels": np.asarray([_type_label(pair[1]) for pair in pairs], dtype=np.int64),
        "user_ids": np.asarray([pair[0] for pair in pairs]),
        "group_names": np.asarray([pair[1] for pair in pairs]),
    }


def _load_model(config, checkpoint_path: Path, modalities: list[str], target_dim: int) -> ArticulatoryStudent:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = ArticulatoryStudent(
        modalities=modalities,
        embedding_dim=int(config["modalities"][modalities[0]].get("embedding_dim", 128)),
        hidden_dim=int(config["student"].get("hidden_dim", 256)),
        bottleneck_dim=int(config["student"].get("bottleneck_dim", 64)),
        target_dim=target_dim,
        num_classes=int(config["classes"]["num_classes"]),
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def _write_report(path: Path, summary: pd.DataFrame) -> None:
    rows = []
    for item in summary.itertuples(index=False):
        rows.append(
            f"| {item.task.replace('_', ' ').title()} | {item.layer.replace('_', ' ').title()} | "
            f"{100 * item.accuracy_mean:.1f}% | {100 * item.accuracy_std:.1f}% | "
            f"{100 * item.chance_accuracy:.1f}% |"
        )
    best_class = summary[summary["task"] == "utterance_class"].sort_values("accuracy_mean", ascending=False).iloc[0]
    best_type = summary[summary["task"] == "utterance_type"].sort_values("accuracy_mean", ascending=False).iloc[0]
    speaker = summary[summary["task"] == "speaker_leakage"].set_index("layer")
    text = f"""# HuBERT Student Interpretability Probes

Frozen linear probes were evaluated across the same five encoder-disjoint folds as the
student CV experiment.

## Probe Design

- Utterance class and type probes use speaker-disjoint train/validation/test splits.
- Speaker leakage probes use training speakers in every split but hold out complete
  utterance classes, testing whether identity generalizes across unseen content.
- Probe regularization is selected on validation data before refitting on train plus
  validation data.
- `Teacher HuBERT` is the real audio target and serves as a reference, not a sensor-only
  inference representation.

## Aggregate Results

| Task | Representation | Mean Accuracy | Std. Dev. | Chance |
|---|---|---:|---:|---:|
{chr(10).join(rows)}

## Main Findings

- The strongest class representation is **{best_class.layer.replace('_', ' ')}** at
  **{100 * best_class.accuracy_mean:.1f}%** mean speaker-disjoint accuracy.
- The strongest utterance-type representation is **{best_type.layer.replace('_', ' ')}**
  at **{100 * best_type.accuracy_mean:.1f}%**.
- Speaker leakage changes from **{100 * speaker.loc['sensor_input', 'accuracy_mean']:.1f}%**
  in the concatenated sensor input to **{100 * speaker.loc['bottleneck', 'accuracy_mean']:.1f}%**
  in the student bottleneck. Higher speaker-probe accuracy means more identity leakage.

These probes measure linear decodability, not causal use. The modality attribution
experiment and later feature ablations are needed to determine which sensor inputs and
features drive the decoded information.
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/real_embeddings.local.yaml")
    parser.add_argument("--teacher-targets", default="artifacts/teacher_targets/facebook_hubert-base-ls960_targets.npz")
    parser.add_argument("--student-dir", default="artifacts/students/hubert_cv")
    parser.add_argument("--activations-dir", default="artifacts/activations/hubert_cv")
    parser.add_argument("--results-output", default="reports/results/hubert_student_probe_results.csv")
    parser.add_argument("--summary-output", default="reports/results/hubert_student_probe_summary.csv")
    parser.add_argument("--report-output", default="reports/hubert_student_probes.md")
    args = parser.parse_args()

    config = load_config(args.config)
    teacher = load_teacher_targets(args.teacher_targets)
    seed_paths, _ = resolve_embedding_paths(config["data"], [config["data"]["embeddings_dir"], ".", "extra", "notebooks"])
    manifest = build_manifest(seed_paths)
    folds = make_speaker_kfold_splits(manifest, config["splits"]["num_folds"], config["project"]["seed"])
    activation_dir = Path(args.activations_dir)
    activation_dir.mkdir(parents=True, exist_ok=True)
    result_rows = []
    total_start = time.perf_counter()

    for position, fold in enumerate(folds, start=1):
        fold_id = int(fold["fold"])
        start = time.perf_counter()
        paths = configured_fold_embedding_paths(config.get("true_encoder_cv", {}), fold_id)
        payloads = {modality: load_embedding_repetitions(path) for modality, path in paths.items()}
        checkpoint_path = Path(args.student_dir) / f"fold_{fold_id}_teacher_student.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        modalities = list(checkpoint["modalities"])
        common_payloads = {modality: payloads[modality] for modality in modalities}
        pairs_by_split = {
            split: common_teacher_pairs(common_payloads, teacher, fold[f"{split}_speakers"])
            for split in ("train", "val", "test")
        }
        arrays = {
            split: _split_arrays(common_payloads, teacher, modalities, pairs)
            for split, pairs in pairs_by_split.items()
        }
        teacher_center = np.asarray(
            checkpoint.get("teacher_center", np.zeros(int(teacher["target_dim"]), dtype=np.float32)),
            dtype=np.float32,
        )
        for split_arrays in arrays.values():
            split_arrays["teacher_hubert"] = split_arrays["teacher_hubert"] - teacher_center
        model = _load_model(config, checkpoint_path, modalities, int(teacher["target_dim"]))
        activations = {}
        save_payload = {}
        for split, split_arrays in arrays.items():
            layers = extract_student_activations(model, split_arrays["inputs"])
            layers["teacher_hubert"] = split_arrays["teacher_hubert"]
            activations[split] = layers
            for layer, values in layers.items():
                save_payload[f"{split}_{layer}"] = values
            for key in ("labels", "type_labels", "user_ids", "group_names"):
                save_payload[f"{split}_{key}"] = split_arrays[key]
        np.savez_compressed(activation_dir / f"fold_{fold_id}_activations.npz", **save_payload)

        for task, label_key in (("utterance_class", "labels"), ("utterance_type", "type_labels")):
            for layer in LAYER_ORDER:
                result = fit_linear_probe(
                    activations["train"][layer], arrays["train"][label_key],
                    activations["val"][layer], arrays["val"][label_key],
                    activations["test"][layer], arrays["test"][label_key],
                    seed=int(config["project"]["seed"]) + fold_id,
                )
                result_rows.append({"fold": fold_id, "task": task, "layer": layer, "chance_accuracy": 1 / result["num_classes"], **result})

        train_class_labels = arrays["train"]["labels"]
        speaker_train, speaker_val, speaker_test = content_heldout_indices(
            train_class_labels, seed=int(config["project"]["seed"]) + fold_id
        )
        speaker_ids = arrays["train"]["user_ids"]
        for layer in LAYER_ORDER:
            values = activations["train"][layer]
            result = fit_linear_probe(
                values[speaker_train], speaker_ids[speaker_train],
                values[speaker_val], speaker_ids[speaker_val],
                values[speaker_test], speaker_ids[speaker_test],
                seed=int(config["project"]["seed"]) + fold_id,
            )
            result_rows.append({"fold": fold_id, "task": "speaker_leakage", "layer": layer, "chance_accuracy": 1 / result["num_classes"], **result})

        elapsed = time.perf_counter() - start
        average = (time.perf_counter() - total_start) / position
        remaining = average * (len(folds) - position)
        print(f"PROBE_PROGRESS fold={fold_id} elapsed_seconds={elapsed:.1f} estimated_remaining_seconds={remaining:.1f}", flush=True)

    results = pd.DataFrame(result_rows)
    summary = (
        results.groupby(["task", "layer"], as_index=False)
        .agg(
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            macro_f1_mean=("macro_f1", "mean"),
            chance_accuracy=("chance_accuracy", "mean"),
            folds=("fold", "nunique"),
        )
    )
    results_path = Path(args.results_output)
    summary_path = Path(args.summary_output)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(results_path, index=False)
    summary.to_csv(summary_path, index=False)
    _write_report(Path(args.report_output), summary)
    print(f"Saved probe results to {results_path}")
    print(f"Saved pushable report to {args.report_output}")


if __name__ == "__main__":
    main()
