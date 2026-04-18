#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
from pathlib import Path

import numpy as np
import librosa


# ========== config (can be moved to YAML later) ==========
SR = 16_000                # target sample rate
N_FFT = 1024
HOP_LENGTH = 256           # ~16 ms hop at 16 kHz
N_MELS = 80

WIN_SEC = 0.5              # late-fusion window length
HOP_SEC = 0.25             # late-fusion window hop

# =========================================================

def load_audio(path, sr=SR):
    y, sr = librosa.load(path, sr=sr, mono=True)
    # guard: trim leading/trailing absolute silence
    y, _ = librosa.effects.trim(y, top_db=40)
    return y, sr

def compute_frame_features(y, sr):
    """Return (F, T) where F = 242 dims, T = #frames."""
    # mel spectrogram -> dB
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)  # (80, T)

    # deltas
    mel_d1 = librosa.feature.delta(mel_db, order=1, width=9)
    mel_d2 = librosa.feature.delta(mel_db, order=2, width=9)

    # rms (1, T)
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)

    # pitch using PYIN aligned to same hop
    f0, vflag, _ = librosa.pyin(
        y, fmin=50, fmax=300, frame_length=N_FFT, hop_length=HOP_LENGTH, sr=sr
    )  # f0 shape (T,), with NaNs for unvoiced

    # interpolate NaNs in f0 to keep a dense vector
    f0 = f0.astype(float)
    t = np.arange(f0.shape[0])
    nan_mask = np.isnan(f0)
    if nan_mask.all():
        f0[:] = 0.0
    else:
        f0[nan_mask] = np.interp(t[nan_mask], t[~nan_mask], f0[~nan_mask])

    # ensure all T match
    T = mel_db.shape[1]
    assert mel_d1.shape[1] == T and mel_d2.shape[1] == T and rms.shape[1] == T and f0.shape[0] == T

    # stack: (F, T)
    feat = np.vstack([
        mel_db,          # 80
        mel_d1,          # 80
        mel_d2,          # 80
        rms,             # 1
        f0.reshape(1, -1)  # 1
    ])
    return feat  # (242, T)

def window_and_aggregate(feat_frame, sr):
    """
    feat_frame: (F, Tframes)
    returns features_windows: (Tw, 2F) with mean & std per window
    """
    frames_per_sec = sr / HOP_LENGTH
    win_frames = int(round(WIN_SEC * frames_per_sec))
    hop_frames = int(round(HOP_SEC * frames_per_sec))

    F, T = feat_frame.shape
    if T < win_frames:
        # pad with last frame to fit one window
        pad = np.repeat(feat_frame[:, -1:], win_frames - T, axis=1)
        feat_frame = np.hstack([feat_frame, pad])
        T = feat_frame.shape[1]

    starts = np.arange(0, T - win_frames + 1, hop_frames, dtype=int)
    out = []
    for s in starts:
        w = feat_frame[:, s:s + win_frames]  # (F, W)
        mu = w.mean(axis=1)
        sd = w.std(axis=1)
        out.append(np.hstack([mu, sd]))  # (2F,)
    features = np.stack(out, axis=0)  # (Tw, 2F)
    meta = {
        "sr": SR,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "n_mels": N_MELS,
        "frame_dim": F,
        "window_sec": WIN_SEC,
        "hop_sec": HOP_SEC,
        "start_ms": 0,
        "hop_ms": int(round(HOP_SEC * 1000)),
        "frames_per_sec": frames_per_sec,
        "notes": "mean+std aggregation of log-mel, deltas, rms, f0"
    }
    return features.astype(np.float32), meta

def save_npz(out_path, features, meta):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, features=features, meta=meta)

def main():
    ap = argparse.ArgumentParser(description="Extract audio features (Mel+Δ+ΔΔ+RMS+F0) → windowed mean/std.")
    ap.add_argument("--audio", required=True, help="Path to input audio file (wav, flac, etc.)")
    ap.add_argument("--outdir", required=True, help="Output directory for NPZ")
    ap.add_argument("--session", default="session001", help="Session ID for naming")
    args = ap.parse_args()

    y, sr = load_audio(args.audio, SR)
    frame_feat = compute_frame_features(y, sr)              # (242, T)
    win_feat, meta = window_and_aggregate(frame_feat, sr)   # (Tw, 484)

    out_path = Path(args.outdir) / f"{args.session}__audio.npz"
    save_npz(out_path, win_feat, meta)

    print(f"[OK] saved {out_path}")
    print(f"shape: {win_feat.shape}  dim_per_window: {win_feat.shape[1]}  windows: {win_feat.shape[0]}")

if __name__ == "__main__":
    main()
