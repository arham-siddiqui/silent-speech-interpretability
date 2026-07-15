#!/usr/bin/env python3
"""Build a central manifest and dataset audit."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silent_speech_interpretability.configs import load_config
from silent_speech_interpretability.data.manifest import (
    build_intersection_manifest,
    build_manifest,
    enforce_quality_gates,
    resolve_embedding_paths,
    write_dataset_audit,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--override", action="append", default=[])
    parser.add_argument("--enforce-gates", action="store_true", help="Fail if dataset quality gates are not satisfied.")
    args = parser.parse_args()

    config = load_config(args.config, args.override)
    embedding_paths, path_sources = resolve_embedding_paths(
        config["data"],
        [config["data"]["embeddings_dir"], ".", "extra", "notebooks"],
    )
    manifest = build_manifest(embedding_paths, synthetic_if_missing=True)
    intersection = build_intersection_manifest(manifest)

    manifest_path = Path(config["data"]["manifest_path"])
    intersection_path = Path(config["data"].get("intersection_manifest_path", "artifacts/manifest_intersection.csv"))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(manifest_path, index=False)
    intersection.to_csv(intersection_path, index=False)

    audit_path = Path(config["data"]["results_dir"]) / "dataset_audit.json"
    audit = write_dataset_audit(manifest, audit_path, embedding_paths, path_sources)
    print(f"Wrote {manifest_path} ({len(manifest)} rows)")
    print(f"Wrote {intersection_path} ({len(intersection)} rows)")
    print(f"Wrote {audit_path}")
    print(f"Counts by modality: {audit['modality_counts']}")
    if "alignment" in audit:
        alignment = audit["alignment"]
        print(f"Strict embedding intersection: {alignment['strict_intersection_group_count']} groups")
        print(f"Label mismatches: {alignment['label_mismatch_count']}")
        print(f"User ID mismatches: {alignment['user_id_mismatch_count']}")
    if args.enforce_gates:
        enforce_quality_gates(audit, config.get("quality_gates", {}))
        print("Quality gates passed")


if __name__ == "__main__":
    main()
