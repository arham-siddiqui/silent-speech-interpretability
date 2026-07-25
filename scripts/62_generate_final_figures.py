#!/usr/bin/env python3
"""Generate final static figures for external generalization and radar ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


COLORS = {
    "session": "#0f766e",
    "subject": "#b45309",
    "all": "#1f2937",
    "s32": "#2563eb",
    "magnitude": "#0f766e",
    "delta": "#d97706",
    "s12": "#7c3aed",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path, preview_dir: Path | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_dir / f"{path.stem}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _summary_row(summary: pd.DataFrame, protocol: str, metric: str) -> pd.Series:
    rows = summary[(summary["protocol"] == protocol) & (summary["metric"] == metric)]
    if len(rows) != 1:
        raise ValueError(f"Expected one {protocol}/{metric} row, found {len(rows)}")
    return rows.iloc[0]


def _generalization_figure(
    summary: pd.DataFrame,
    output: Path,
    preview_dir: Path | None,
) -> None:
    protocols = ["session", "subject"]
    labels = ["Held-out session", "Held-out subject"]
    accuracy = [_summary_row(summary, protocol, "accuracy") for protocol in protocols]
    order = [_summary_row(summary, protocol, "order_margin_reversed") for protocol in protocols]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    positions = np.arange(len(protocols))
    values = np.asarray([row["mean"] for row in accuracy])
    lower = values - np.asarray([row["unit_bootstrap_95_low"] for row in accuracy])
    upper = np.asarray([row["unit_bootstrap_95_high"] for row in accuracy]) - values
    axes[0].bar(
        positions,
        100 * values,
        color=[COLORS[protocol] for protocol in protocols],
        width=0.58,
        yerr=100 * np.stack([lower, upper]),
        capsize=5,
    )
    axes[0].axhline(2.0, color="#6b7280", linestyle="--", linewidth=1.2)
    axes[0].text(1.46, 2.8, "2% chance", ha="right", color="#4b5563", fontsize=9)
    axes[0].set_xticks(positions, labels)
    axes[0].set_ylabel("50-class accuracy (%)")
    axes[0].set_ylim(0, max(42, 100 * (values + upper).max() + 4))
    axes[0].set_title("Generalization collapses across speakers")
    for position, value in zip(positions, values, strict=True):
        axes[0].text(position, 100 * value + 1.1, f"{100 * value:.1f}%", ha="center", weight="bold")

    order_values = np.asarray([row["mean"] for row in order])
    order_lower = order_values - np.asarray([row["unit_bootstrap_95_low"] for row in order])
    order_upper = np.asarray([row["unit_bootstrap_95_high"] for row in order]) - order_values
    axes[1].bar(
        positions,
        order_values,
        color=[COLORS[protocol] for protocol in protocols],
        width=0.58,
        yerr=np.stack([order_lower, order_upper]),
        capsize=5,
    )
    axes[1].axhline(0, color="#6b7280", linewidth=1.2)
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("True-minus-reversed HuBERT cosine")
    axes[1].set_ylim(-0.08, 0.44)
    axes[1].set_title("Ordered alignment is session-specific")
    for position, value in zip(positions, order_values, strict=True):
        offset = 0.018 if value >= 0 else -0.034
        axes[1].text(position, value + offset, f"{value:+.3f}", ha="center", weight="bold")

    fig.suptitle("External radar generalization", fontsize=16, fontweight="bold", y=1.03)
    fig.tight_layout()
    _save(fig, output, preview_dir)


def _ablation_figure(
    ablations: pd.DataFrame,
    output: Path,
    preview_dir: Path | None,
) -> None:
    summary = (
        ablations.groupby("feature_set")
        .agg(
            accuracy=("accuracy", "mean"),
            cosine=("segment_cosine", "mean"),
            order_margin=("order_margin_reversed", "mean"),
        )
        .reset_index()
        .sort_values("accuracy")
    )
    labels = {
        "all": "S12 + S32, magnitude + delta",
        "s32": "S32 only",
        "magnitude": "Magnitude only",
        "delta": "Temporal delta only",
        "s12": "S12 only",
    }
    positions = np.arange(len(summary))
    colors = [COLORS[value] for value in summary["feature_set"]]

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.3), sharey=True)
    axes[0].barh(positions, 100 * summary["accuracy"], color=colors, height=0.62)
    axes[0].axvline(2.0, color="#6b7280", linestyle="--", linewidth=1.2)
    axes[0].set_yticks(positions, [labels[value] for value in summary["feature_set"]])
    axes[0].set_xlabel("50-class accuracy (%)")
    axes[0].set_title("Command decoding")
    axes[0].set_xlim(0, 34)
    for position, value in zip(positions, summary["accuracy"], strict=True):
        axes[0].text(100 * value + 0.6, position, f"{100 * value:.1f}%", va="center")

    axes[1].barh(positions, summary["cosine"], color=colors, height=0.62)
    axes[1].set_xlabel("True-order HuBERT cosine")
    axes[1].set_title("Audio-teacher alignment")
    axes[1].set_xlim(0, 0.36)
    axes[1].tick_params(axis="y", labelleft=False)
    for position, value in zip(positions, summary["cosine"], strict=True):
        axes[1].text(value + 0.008, position, f"{value:.3f}", va="center")

    fig.suptitle("External radar feature ablations", fontsize=16, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save(fig, output, preview_dir)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default="reports/tables/external_radar_generalization_summary.csv",
    )
    parser.add_argument(
        "--ablations",
        default="reports/tables/external_radar_feature_ablation.csv",
    )
    parser.add_argument("--figures-dir", default="reports/figures")
    parser.add_argument("--preview-dir", default=None)
    args = parser.parse_args()

    _style()
    figures_dir = Path(args.figures_dir)
    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    _generalization_figure(
        pd.read_csv(args.summary),
        figures_dir / "final_external_generalization.svg",
        preview_dir,
    )
    _ablation_figure(
        pd.read_csv(args.ablations),
        figures_dir / "final_external_feature_ablation.svg",
        preview_dir,
    )
    print(f"Saved final figures to {figures_dir}")


if __name__ == "__main__":
    main()
