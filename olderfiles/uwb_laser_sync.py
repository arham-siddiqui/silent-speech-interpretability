"""
UWB–Laser Synchronization Analysis

This script synchronizes **UWB radar** amplitudes and laser vibration frequencies
for sentence data and provides visualization tools to check synchronization.

Author: Generated for SilentSpeech project
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import warnings
warnings.filterwarnings("ignore")


class UWBLaserSyncAnalyzer:
    """
    Analyze synchronization between **UWB radar** amplitudes and laser vibration frequencies.
    Directory layout assumed (customize in __init__ if needed):

    data_root/
      ├─ uwb_processed/<subject>/sentences<sentence>/...
      └─ laser_processed/<subject>/sentences<sentence>/Laser_data_person_sample{sample}.npy
    """

    def __init__(self, data_root: str = "src/data/rvtall/processed_cut_data"):
        """
        Args:
            data_root: Root path to the processed data directory
        """
        self.data_root = Path(data_root)
        self.uwb_path = self.data_root / "uwb_processed"
        self.laser_path = self.data_root / "laser_processed"

    # --------------------
    # Loading helpers
    # --------------------
    def _try_paths(self, candidates: List[Path]) -> Optional[Path]:
        for p in candidates:
            if p.exists():
                return p
        return None

    def load_uwb_data(self, subject_id: int, sentence_id: int, sample_id: int) -> np.ndarray:
        """
        Load UWB data for a specific subject, sentence, and sample.

        Tries a few common file-name patterns; if none match, it will glob for first .npy.

        Args:
            subject_id: Subject identifier (1-20)
            sentence_id: Sentence identifier (1-10)
            sample_id: Sample identifier (1-7)

        Returns:
            UWB amplitude (or multichannel) data as a numpy array
        """
        base = self.uwb_path / str(subject_id) / f"sentences_{sentence_id}"

        # Common filename patterns we’ve seen in similar datasets
        candidates = [
            base / f"{subject_id}_uwb_{sentence_id}_sample{sample_id}.npy",
            base / f"{subject_id}_UWB_{sentence_id}_sample{sample_id}.npy",
            base / f"{subject_id}_3_{sentence_id}_sample{sample_id}.npy",  # same as FMCW naming, just in uwb_processed
            base / f"uwb_sample{sample_id}.npy",
        ]
        found = self._try_paths(candidates)

        if found is None:
            # Fallback: pick the first .npy that contains the sample id in its name
            g = list(base.glob(f"*sample{sample_id}*.npy"))
            if not g:
                # Final fallback: any .npy in the folder
                g = list(base.glob("*.npy"))
            if not g:
                raise FileNotFoundError(f"No UWB .npy found under: {base}")
            found = sorted(g)[0]

        return np.load(found)

    def load_laser_data(self, subject_id: int, sentence_id: int, sample_id: int) -> np.ndarray:
        """
        Load laser data.

        Args:
            subject_id: Subject identifier (1-20)
            sentence_id: Sentence identifier (1-10)
            sample_id: Sample identifier (4-7, as per data structure)

        Returns:
            Laser vibration frequency data as numpy array
        """
        laser_file = (
            self.laser_path
            / str(subject_id)
            / f"sentences{sentence_id}"
            / f"Laser_data_person_sample{sample_id}.npy"
        )
        if not laser_file.exists():
            # Basic fallback: first npy containing sample id
            g = list(laser_file.parent.glob(f"*sample{sample_id}*.npy"))
            if not g:
                g = list(laser_file.parent.glob("*.npy"))
            if not g:
                raise FileNotFoundError(f"Laser data file not found: {laser_file}")
            laser_file = sorted(g)[0]
        return np.load(laser_file)

    # --------------------
    # Time / preprocessing
    # --------------------
    def create_time_frames(
        self,
        uwb_data: np.ndarray,
        laser_data: np.ndarray,
        uwb_sampling_rate: float = 300.0,
        laser_sampling_rate: float = 1470.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create time bases for UWB and Laser.

        Returns:
            (uwb_time, laser_time)
        """
        # If UWB is multi-channel, time is the first dimension
        uwb_time = np.arange(uwb_data.shape[0], dtype=float) / float(uwb_sampling_rate)
        laser_time = np.arange(len(laser_data), dtype=float) / float(laser_sampling_rate)
        return uwb_time, laser_time

    def extract_uwb_amplitudes(self, uwb_data: np.ndarray) -> np.ndarray:
        """
        Extract a 1-D amplitude from UWB data.
        Default: mean across channels (axis=1) if multi-channel; otherwise return as-is.
        """
        return uwb_data.mean(axis=1) if uwb_data.ndim > 1 else uwb_data

    # --------------------
    # Plotting
    # --------------------
    def plot_uwb_amplitudes(self, uwb_amp: np.ndarray, time: np.ndarray, title: str) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(time, uwb_amp, linewidth=1, alpha=0.9)
        plt.xlabel("Time (s)")
        plt.ylabel("UWB Amplitude")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_laser_frequencies(self, laser_data: np.ndarray, time: np.ndarray, title: str) -> None:
        plt.figure(figsize=(12, 6))
        plt.plot(time, laser_data, linewidth=1, alpha=0.9)
        plt.xlabel("Time (s)")
        plt.ylabel("Laser Vibration Frequency")
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_synchronized_data(
        self,
        uwb_amp: np.ndarray,
        laser_data: np.ndarray,
        uwb_time: np.ndarray,
        laser_time: np.ndarray,
        title: str = "Synchronized UWB and Laser Data",
    ) -> None:
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 12))

        ax1.plot(uwb_time, uwb_amp, linewidth=1, alpha=0.9, label="UWB Amplitude")
        ax1.set_ylabel("UWB Amplitude")
        ax1.set_title("UWB Amplitudes")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(laser_time, laser_data, linewidth=1, alpha=0.9, label="Laser Frequencies")
        ax2.set_ylabel("Laser Vibration Frequency")
        ax2.set_title("Laser Vibration Frequencies")
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Normalize and overlay
        def _norm(x: np.ndarray) -> np.ndarray:
            x = x.astype(float)
            rng = x.max() - x.min()
            return (x - x.min()) / (rng if rng != 0 else 1.0)

        ax3.plot(uwb_time, _norm(uwb_amp), linewidth=1, alpha=0.9, label="UWB (normalized)")
        ax3.plot(laser_time, _norm(laser_data), linewidth=1, alpha=0.9, label="Laser (normalized)")
        ax3.set_xlabel("Time (s)")
        ax3.set_ylabel("Normalized Amplitude")
        ax3.set_title("Overlaid UWB and Laser (Normalized)")
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        plt.show()

    # --------------------
    # Top-level analysis
    # --------------------
    def analyze_synchronization(
        self,
        subject_id: int,
        sentence_id: int,
        uwb_sample_id: int = 1,
        laser_sample_id: int = 5,
        uwb_fs: float = 300.0,
        laser_fs: float = 1470.0,
    ) -> Optional[Dict]:
        """
        Perform synchronization analysis for a given subject/sentence (UWB vs Laser).
        Mirrors your radar version so you can swap easily.

        Returns:
            Dict with arrays & stats, or None on error.
        """
        print(f"Analyzing UWB–Laser synchronization | Subject {subject_id}, Sentence {sentence_id}")
        print(f"UWB Sample: {uwb_sample_id}, Laser Sample: {laser_sample_id}")

        try:
            uwb_raw = self.load_uwb_data(subject_id, sentence_id, uwb_sample_id)
            laser = self.load_laser_data(subject_id, sentence_id, laser_sample_id)

            print(f"UWB data shape: {uwb_raw.shape}")
            print(f"Laser data shape: {laser.shape}")

            uwb_amp = self.extract_uwb_amplitudes(uwb_raw)
            uwb_time, laser_time = self.create_time_frames(uwb_raw, laser, uwb_fs, laser_fs)

            print(f"UWB time range: {uwb_time[0]:.3f}s → {uwb_time[-1]:.3f}s")
            print(f"Laser time range: {laser_time[0]:.3f}s → {laser_time[-1]:.3f}s")

            # Plots
            self.plot_uwb_amplitudes(
                uwb_amp, uwb_time, f"Subject {subject_id}, Sentence {sentence_id} – UWB Amplitudes"
            )
            self.plot_laser_frequencies(
                laser, laser_time, f"Subject {subject_id}, Sentence {sentence_id} – Laser Frequencies"
            )
            self.plot_synchronized_data(
                uwb_amp, laser, uwb_time, laser_time,
                f"Subject {subject_id}, Sentence {sentence_id} – UWB vs Laser (Overlay)"
            )

            uwb_stats = {
                "mean": float(np.mean(uwb_amp)),
                "std": float(np.std(uwb_amp)),
                "min": float(np.min(uwb_amp)),
                "max": float(np.max(uwb_amp)),
            }
            laser_stats = {
                "mean": float(np.mean(laser)),
                "std": float(np.std(laser)),
                "min": float(np.min(laser)),
                "max": float(np.max(laser)),
            }

            print("\nUWB stats:", uwb_stats)
            print("Laser stats:", laser_stats)

            return {
                "uwb_raw": uwb_raw,
                "laser_data": laser,
                "uwb_amplitudes": uwb_amp,
                "uwb_time": uwb_time,
                "laser_time": laser_time,
                "uwb_stats": uwb_stats,
                "laser_stats": laser_stats,
            }
        except Exception as e:
            print(f"Error in UWB analysis: {e}")
            return None


def main():
    """
    Minimal demo runner (mirrors your radar script’s main()).
    """
    analyzer = UWBLaserSyncAnalyzer()

    print("=" * 60)
    print("UWB–LASER SYNCHRONIZATION ANALYSIS")
    print("=" * 60)

    # Tweak these IDs to your dataset
    subject_id = 1
    sentence_id = 5
    uwb_sample_id = 1
    laser_sample_id = 5

    results = analyzer.analyze_synchronization(
        subject_id=subject_id,
        sentence_id=sentence_id,
        uwb_sample_id=uwb_sample_id,
        laser_sample_id=laser_sample_id,
        uwb_fs=300.0,
        laser_fs=1470.0,
    )

    if results:
        print("\nAnalysis completed successfully (UWB).")
    else:
        print("Analysis failed. Verify paths/file names for UWB/laser.")


if __name__ == "__main__":
    main()
