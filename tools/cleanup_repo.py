#!/usr/bin/env python3
"""Compatibility wrapper for scripts/00_cleanup_repo.py."""

from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts" / "00_cleanup_repo.py"), run_name="__main__")
