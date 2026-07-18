#!/usr/bin/env python3
"""Extract four-segment activations from fold-specific silent-sensor encoders."""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from silent_speech_interpretability.interp.temporal import lip_articulation_segments, pool_temporal_segments
from silent_speech_interpretability.models.encoders.laser import (
    LaserCNNLSTMEncoder,
    LaserDataset,
    build_laser_sample_list,
    collate_laser_batch,
)
from silent_speech_interpretability.models.encoders.lip import (
    LipExtractionDataset,
    LipLSTMV2,
    build_lip_sample_list,
    collate_lip_extraction_batch,
)
from silent_speech_interpretability.models.encoders.mmwave import (
    MmwaveCNNLSTMEncoder,
    MmwaveDataset,
    build_mmwave_sample_list,
    collate_mmwave_batch,
)
from silent_speech_interpretability.models.encoders.uwb import (
    UWBDataset,
    UWBEncoderV2,
    build_uwb_sample_list,
    collate_uwb_batch,
)


MODALITIES = ("lip", "laser", "mmwave", "uwb")


def _device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _label_map(fold_dir: Path, modality: str) -> dict[str, int]:
    return {str(key): int(value) for key, value in json.loads((fold_dir / f"{modality}_label_map.json").read_text()).items()}


def _model(fold_dir: Path, modality: str, metadata: dict, label_map: dict[str, int], device: torch.device):
    config = metadata[f"{modality}_training"]
    num_classes = len(label_map)
    if modality == "lip":
        model = LipLSTMV2(
            input_size=80,
            num_classes=num_classes,
            num_speakers=len(metadata["train_speakers"]),
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            embedding_dim=int(config["embedding_dim"]),
            dropout=float(config["dropout"]),
        )
        checkpoint = fold_dir / "lip_lstm_model.pt"
    elif modality == "laser":
        model = LaserCNNLSTMEncoder(
            num_classes=num_classes,
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            embedding_dim=int(config["embedding_dim"]),
            dropout=float(config["dropout"]),
        )
        checkpoint = fold_dir / "laser_cnn_lstm_model.pt"
    elif modality == "mmwave":
        model = MmwaveCNNLSTMEncoder(
            num_classes=num_classes,
            hidden_size=int(config["hidden_size"]),
            num_layers=int(config["num_layers"]),
            embedding_dim=int(config["embedding_dim"]),
            dropout=float(config["dropout"]),
        )
        checkpoint = fold_dir / "mmwave_cnn_lstm_model.pt"
    elif modality == "uwb":
        model = UWBEncoderV2(
            num_classes=num_classes,
            num_speakers=len(metadata["train_speakers"]),
            hidden_size=int(config["hidden_size"]),
            embedding_dim=int(config["embedding_dim"]),
            dropout=float(config["dropout"]),
        )
        checkpoint = fold_dir / "uwb_cnn_lstm_model.pt"
    else:
        raise ValueError(modality)
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def _samples_and_loader(modality: str, samples: list[dict[str, str]], label_map: dict[str, int], batch_size: int):
    if modality == "lip":
        dataset = LipExtractionDataset(samples, label_map)
        collate = collate_lip_extraction_batch
    elif modality == "laser":
        dataset = LaserDataset(samples, label_map, augment=False)
        collate = collate_laser_batch
    elif modality == "mmwave":
        dataset = MmwaveDataset(samples, label_map, augment=False)
        collate = collate_mmwave_batch
    else:
        dataset = UWBDataset(samples, label_map, {}, augment=False)
        collate = collate_uwb_batch
    return dataset, DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)


def _grouped_payload(
    records: dict[tuple[str, str], list[np.ndarray]],
    labels: dict[tuple[str, str], int],
) -> dict[str, np.ndarray]:
    pairs = sorted(records)
    return {
        "values": np.stack([np.mean(records[pair], axis=0) for pair in pairs]).astype(np.float32),
        "user_ids": np.asarray([pair[0] for pair in pairs]),
        "group_names": np.asarray([pair[1] for pair in pairs]),
        "labels": np.asarray([labels[pair] for pair in pairs], dtype=np.int64),
        "repetition_counts": np.asarray([len(records[pair]) for pair in pairs], dtype=np.int64),
    }


def _extract_modality(
    model,
    modality: str,
    loader: DataLoader,
    device: torch.device,
    num_segments: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None, dict[str, float]]:
    records: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    articulation: dict[tuple[str, str], list[np.ndarray]] = defaultdict(list)
    labels: dict[tuple[str, str], int] = {}
    timestep_counts = []
    started = time.perf_counter()
    with torch.no_grad():
        for batch in loader:
            if modality == "lip":
                padded, lengths, batch_labels, samples = batch
                raw_lip = padded[:, :, :40].numpy()
            elif modality in {"laser", "mmwave"}:
                padded, lengths, batch_labels, samples = batch
                raw_lip = None
            else:
                padded, lengths, batch_labels, _speakers, samples = batch
                raw_lip = None
            sequence, output_lengths = model.encode_sequence(padded.to(device), lengths.to(device))
            segments = pool_temporal_segments(sequence, output_lengths, num_segments).cpu().numpy()
            timestep_counts.extend(output_lengths.cpu().numpy().tolist())
            for index, sample in enumerate(samples):
                pair = (str(sample["user_id"]), str(sample["group_name"]))
                records[pair].append(segments[index])
                labels[pair] = int(batch_labels[index])
                if raw_lip is not None:
                    articulation[pair].append(lip_articulation_segments(raw_lip[index, : int(lengths[index])], num_segments))
    payload = _grouped_payload(records, labels)
    articulation_payload = _grouped_payload(articulation, labels) if articulation else None
    audit = {
        "raw_repetitions": float(sum(len(values) for values in records.values())),
        "pairs": float(len(records)),
        "mean_encoder_steps": float(np.mean(timestep_counts)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return payload, articulation_payload, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rvtall-base", default="/Users/arhamsiddiqui/Desktop/silentSpeech/src/data/RVTALL/Processed_cut_data")
    parser.add_argument("--fold-root", default="artifacts/embeddings/speaker_cv")
    parser.add_argument("--output-dir", default="artifacts/activations/temporal_sensors")
    parser.add_argument("--audit-output", default="reports/results/temporal_sensor_activation_audit.csv")
    parser.add_argument("--folds", default="0,1,2,3,4")
    parser.add_argument("--modalities", default=",".join(MODALITIES))
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-pairs", type=int, default=None)
    args = parser.parse_args()

    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    modalities = [value.strip() for value in args.modalities.split(",") if value.strip()]
    builders = {
        "lip": build_lip_sample_list,
        "laser": build_laser_sample_list,
        "mmwave": build_mmwave_sample_list,
        "uwb": build_uwb_sample_list,
    }
    samples_by_modality = {modality: builders[modality](args.rvtall_base) for modality in modalities}
    selected_pairs = None
    if args.limit_pairs is not None:
        all_pairs = sorted({(str(sample["user_id"]), str(sample["group_name"])) for sample in samples_by_modality[modalities[0]]})
        selected_pairs = set(all_pairs[: args.limit_pairs])
        samples_by_modality = {
            modality: [sample for sample in samples if (str(sample["user_id"]), str(sample["group_name"])) in selected_pairs]
            for modality, samples in samples_by_modality.items()
        }

    device = _device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    batch_sizes = {"lip": 64, "laser": 64, "mmwave": 16, "uwb": 16}
    total_started = time.perf_counter()
    for fold_position, fold in enumerate(folds, start=1):
        fold_dir = Path(args.fold_root) / f"fold_{fold}"
        metadata = json.loads((fold_dir / "metadata.json").read_text())
        save_payload: dict[str, np.ndarray] = {"num_segments": np.asarray(args.segments)}
        for modality in modalities:
            label_map = _label_map(fold_dir, modality)
            model = _model(fold_dir, modality, metadata, label_map, device)
            dataset, loader = _samples_and_loader(modality, samples_by_modality[modality], label_map, batch_sizes[modality])
            payload, articulation, audit = _extract_modality(model, modality, loader, device, args.segments)
            for key, values in payload.items():
                save_payload[f"{modality}_{key}"] = values
            if articulation is not None:
                for key, values in articulation.items():
                    save_payload[f"articulation_{key}"] = values
            audit_rows.append({"fold": fold, "modality": modality, "dataset_rows": len(dataset), **audit})
            print(
                f"TEMPORAL_SENSOR fold={fold} modality={modality} pairs={int(audit['pairs'])} "
                f"elapsed_seconds={audit['elapsed_seconds']:.1f}",
                flush=True,
            )
            del model
            if device.type == "mps":
                torch.mps.empty_cache()
        np.savez_compressed(output_dir / f"fold_{fold}_temporal_sensors.npz", **save_payload)
        average = (time.perf_counter() - total_started) / fold_position
        remaining = average * (len(folds) - fold_position)
        print(f"TEMPORAL_SENSOR_FOLD fold={fold} estimated_remaining_seconds={remaining:.1f}", flush=True)

    audit_path = Path(args.audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audit_rows).to_csv(audit_path, index=False)
    print(f"Saved temporal sensor activations to {output_dir}")


if __name__ == "__main__":
    main()
