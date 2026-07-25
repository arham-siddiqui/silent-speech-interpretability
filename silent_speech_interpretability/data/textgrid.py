"""Minimal Praat TextGrid interchange for boundary annotation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable


@dataclass(frozen=True)
class TextGridInterval:
    start: float
    end: float
    text: str


def _quote(value: str) -> str:
    return value.replace('"', '""')


def _contiguous(
    intervals: Iterable[TextGridInterval],
    xmin: float,
    xmax: float,
) -> list[TextGridInterval]:
    ordered = sorted(intervals, key=lambda item: (item.start, item.end))
    output: list[TextGridInterval] = []
    cursor = xmin
    for interval in ordered:
        start = max(xmin, min(float(interval.start), xmax))
        end = max(start, min(float(interval.end), xmax))
        if start > cursor + 1e-9:
            output.append(TextGridInterval(cursor, start, ""))
        if start < cursor - 1e-9:
            raise ValueError("TextGrid intervals overlap")
        if end > start:
            output.append(TextGridInterval(start, end, str(interval.text)))
            cursor = end
    if cursor < xmax - 1e-9:
        output.append(TextGridInterval(cursor, xmax, ""))
    return output


def write_textgrid(
    path: str | Path,
    duration: float,
    tiers: dict[str, Iterable[TextGridInterval]],
) -> None:
    """Write long-form IntervalTier data accepted by Praat."""
    if duration <= 0:
        raise ValueError("TextGrid duration must be positive")
    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "xmin = 0",
        f"xmax = {duration:.9f}",
        "tiers? <exists>",
        "size = " + str(len(tiers)),
        "item []:",
    ]
    for tier_index, (name, raw_intervals) in enumerate(tiers.items(), start=1):
        intervals = _contiguous(raw_intervals, 0.0, duration)
        lines.extend(
            [
                f"    item [{tier_index}]:",
                '        class = "IntervalTier"',
                f'        name = "{_quote(name)}"',
                "        xmin = 0",
                f"        xmax = {duration:.9f}",
                f"        intervals: size = {len(intervals)}",
            ]
        )
        for interval_index, interval in enumerate(intervals, start=1):
            lines.extend(
                [
                    f"        intervals [{interval_index}]:",
                    f"            xmin = {interval.start:.9f}",
                    f"            xmax = {interval.end:.9f}",
                    f'            text = "{_quote(interval.text)}"',
                ]
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


_ASSIGNMENT = re.compile(r"^\s*(name|xmin|xmax|text)\s*=\s*(.*?)\s*$")


def read_textgrid(path: str | Path) -> dict[str, list[TextGridInterval]]:
    """Read IntervalTiers from a long-form TextGrid written by this project or Praat."""
    tiers: dict[str, list[TextGridInterval]] = {}
    tier_name: str | None = None
    interval: dict[str, str] | None = None

    def unquote(value: str) -> str:
        if len(value) >= 2 and value[0] == value[-1] == '"':
            return value[1:-1].replace('""', '"')
        return value

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if re.fullmatch(r"item \[\d+\]:", stripped):
            tier_name = None
            interval = None
            continue
        if re.fullmatch(r"intervals \[\d+\]:", stripped):
            interval = {}
            continue
        match = _ASSIGNMENT.match(line)
        if not match:
            continue
        key, raw_value = match.groups()
        value = unquote(raw_value)
        if interval is None:
            if key == "name":
                tier_name = value
                tiers.setdefault(tier_name, [])
            continue
        interval[key] = value
        if key == "text":
            if tier_name is None or "xmin" not in interval or "xmax" not in interval:
                raise ValueError(f"Malformed interval in {path}")
            if value:
                tiers[tier_name].append(
                    TextGridInterval(float(interval["xmin"]), float(interval["xmax"]), value)
                )
            interval = None
    if not tiers:
        raise ValueError(f"No IntervalTiers found in {path}")
    return tiers
