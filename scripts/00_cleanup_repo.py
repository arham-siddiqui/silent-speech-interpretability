#!/usr/bin/env python3
"""Create the research-grade directory layout and migration report.

This script is intentionally conservative: it creates directories, updates safe
metadata, and only moves folders that are known legacy containers when requested.
It never deletes data, checkpoints, embeddings, or scripts.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "migration_report.md"

PACKAGE_DIRS = [
    "silent_speech_interpretability/configs",
    "silent_speech_interpretability/data",
    "silent_speech_interpretability/models/encoders",
    "silent_speech_interpretability/models/fusion",
    "silent_speech_interpretability/models/teachers",
    "silent_speech_interpretability/models/students",
    "silent_speech_interpretability/training",
    "silent_speech_interpretability/evals",
    "silent_speech_interpretability/interp",
    "silent_speech_interpretability/viz",
    "scripts",
    "tests",
    "reports/results",
    "reports/figures/generated",
    "artifacts",
    "legacy",
    "tools",
]

LEGACY_CANDIDATES = ["olderfiles"]

GITIGNORE_BLOCK = """
# Python
__pycache__/
*.py[cod]
.pytest_cache/
.mypy_cache/
.ruff_cache/

# Virtual environments
venv/
.venv/
env/

# Data
data/
datasets/
raw_data/
processed_data/
src/data/RVTALL/
*.h5
*.hdf5
*.npy
*.npz
*.pt
*.pth
*.ckpt
*.pkl
*.pickle

# Generated artifacts
artifacts/
outputs/
runs/
wandb/
lightning_logs/
reports/results/
reports/figures/generated/

# OS/editor
.DS_Store
.vscode/
.idea/
"""


def ensure_dirs() -> None:
    for rel in PACKAGE_DIRS:
        path = ROOT / rel
        path.mkdir(parents=True, exist_ok=True)
        if rel.startswith("silent_speech_interpretability"):
            init_file = path / "__init__.py"
            init_file.touch(exist_ok=True)


def update_gitignore() -> None:
    path = ROOT / ".gitignore"
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    existing_lines = set(existing.splitlines())
    additions = []
    for line in GITIGNORE_BLOCK.strip().splitlines():
        if line and line not in existing_lines:
            additions.append(line)
    if additions:
        with path.open("a", encoding="utf-8") as f:
            f.write("\n# silent-speech-interpretability generated ignores\n")
            f.write("\n".join(additions))
            f.write("\n")


def move_legacy_candidates(apply: bool) -> list[str]:
    moved = []
    for candidate in LEGACY_CANDIDATES:
        src = ROOT / candidate
        dst = ROOT / "legacy" / candidate
        if not src.exists() or dst.exists():
            continue
        if apply:
            shutil.move(str(src), str(dst))
            moved.append(f"{candidate} -> legacy/{candidate}")
        else:
            moved.append(f"{candidate} -> legacy/{candidate} (dry run)")
    return moved


def write_report(moved: list[str], apply: bool) -> None:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    kept = [
        "src/ legacy-compatible training and model utilities",
        "notebooks/ experimental encoder and fusion scripts",
        "figures/ existing hand-made result figures",
        "extra/ label maps and result metadata",
        "README.md and AI_README.md",
    ]
    newly_added = [
        "silent_speech_interpretability/ package skeleton",
        "configs/defaults.yaml and companion config files",
        "scripts/00_cleanup_repo.py",
        "reports/ migration and results directories",
        "tests/ synthetic fixture tests",
    ]
    text = [
        "# Migration Report",
        "",
        "## Files kept",
        *[f"- {item}" for item in kept],
        "",
        "## Files moved to legacy/",
        *([f"- {item}" for item in moved] if moved else ["- None"]),
        "",
        "## Files replaced",
        "- None; old source files were preserved.",
        "",
        "## Files newly added",
        *[f"- {item}" for item in newly_added],
        "",
        "## Assumptions made",
        "- Python import package uses `silent_speech_interpretability` because hyphens are not valid in package names.",
        "- Existing data, embeddings, checkpoints, figures, and notebooks should be preserved unless explicitly migrated.",
        f"- Cleanup script was run in {'apply' if apply else 'dry-run'} mode.",
        "",
    ]
    REPORT.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="Move known legacy candidates into legacy/.")
    args = parser.parse_args()

    ensure_dirs()
    update_gitignore()
    moved = move_legacy_candidates(args.apply)
    write_report(moved, args.apply)
    print(f"Wrote {REPORT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
