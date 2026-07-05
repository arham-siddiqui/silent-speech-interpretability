"""YAML configuration loading for silent-speech-interpretability."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "defaults.yaml"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge dictionaries without mutating inputs."""
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _parse_scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"none", "null"}:
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    cursor = config
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value


def apply_cli_overrides(config: dict[str, Any], overrides: Iterable[str] | None) -> dict[str, Any]:
    """Apply simple KEY=VALUE overrides, where KEY may be dotted."""
    updated = deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be KEY=VALUE, got {item!r}")
        key, raw_value = item.split("=", 1)
        _set_dotted(updated, key, _parse_scalar(raw_value))
    return updated


def validate_config(config: dict[str, Any]) -> None:
    required = [
        ("project", "name"),
        ("project", "seed"),
        ("data", "manifest_path"),
        ("classes", "num_classes"),
        ("splits", "strategy"),
        ("modalities",),
    ]
    missing = []
    for path in required:
        cursor: Any = config
        for key in path:
            if not isinstance(cursor, dict) or key not in cursor:
                missing.append(".".join(path))
                break
            cursor = cursor[key]
    if missing:
        raise ValueError(f"Missing required config fields: {', '.join(missing)}")


def load_config(
    config_path: str | Path | None = None,
    overrides: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Load defaults, merge a user YAML file, then apply KEY=VALUE overrides."""
    with DEFAULT_CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    if config_path:
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as f:
            config = deep_merge(config, yaml.safe_load(f) or {})

    config = apply_cli_overrides(config, overrides)
    validate_config(config)
    return config
