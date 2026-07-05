# Migration Report

## Files kept
- src/ legacy-compatible training and model utilities
- notebooks/ experimental encoder and fusion scripts
- figures/ existing hand-made result figures
- extra/ label maps and result metadata
- README.md and AI_README.md

## Files moved to legacy/
- olderfiles -> legacy/olderfiles (dry run)

## Files replaced
- None; old source files were preserved.

## Files newly added
- silent_speech_interpretability/ package skeleton
- configs/defaults.yaml and companion config files
- scripts/00_cleanup_repo.py
- reports/ migration and results directories
- tests/ synthetic fixture tests

## Assumptions made
- Python import package uses `silent_speech_interpretability` because hyphens are not valid in package names.
- Existing data, embeddings, checkpoints, figures, and notebooks should be preserved unless explicitly migrated.
- Cleanup script was run in dry-run mode.
