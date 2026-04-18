# notebooks/align_help_run_no_scipy.py
from pathlib import Path
from datetime import datetime
import json
import numpy as np
import matplotlib.pyplot as plt

# ---------- tiny utils (no scipy) ----------

def _sorted_jsons(json_dir: Path):
    return sorted(Path(json_dir).glob("*.json"))

def _duration_seconds(start_str, end_str, fmt="%Y-%m-%d %H:%M:%S.%f"):
    s = datetime.strptime(start_str, fmt)
    e = datetime.strptime(end_str, fmt)
    return (e - s).total_seconds()

def _lin_resample(values: np.ndarray, t_old: np.ndarray, t_new: np.ndarray) -> np.ndarray:
    """Linear resample values(t_old) → values(t_new) using np.interp"""
    # np.interp needs ascending t; ensure float64
    t_old = np.asarray(t_old, dtype=float)
    values = np.asarray(values, dtype=float)
    t_new = np.asarray(t_new, dtype=float)
    # clamp outside to endpoints
    return np.interp(t_new, t_old, values, left=values[0], right=values[-1])

def _xcorr_lag_ms(x: np.ndarray, y: np.ndarray, fs: float) -> float:
    """
    Cross-correlation using numpy (full). Positive lag => x lags y.
    Returns lag in milliseconds.
    """
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    corr = np.correlate(x, y, mode="full")       # length = nx + ny - 1
    lags = np.arange(-len(y)+1, len(x))          # in samples
    k = int(np.argmax(corr))
    lag_samp = int(lags[k])
    return 1000.0 * lag_samp / max(fs, 1e-9), lag_samp

def _norm(a: np.ndarray) -> np.ndarray:
    a = a.astype(float)
    r = a.max() - a.min()
    return (a - a.min()) / (r if r > 0 else 1.0)

# ---------- main helper ----------

def align_uwb_laser_by_json_no_scipy(
    uwb_dir,            # folder with UWB ...sampleN.npy (spectrogram patch: freq x time)
    laser_dir,          # folder with Laser ...sampleN.npy (1-D or 2-D)
    json_dir,           # Kinect timestamps folder used during cutting
    sample_id: int,
    reduce_fn: str = "sum",    # "sum" | "mean"  (or pass a callable(freq x T)->(T,))
    refine=True,               # xcorr micro-align
):
    uwb_dir  = Path(uwb_dir)
    laser_dir = Path(laser_dir)
    json_dir = Path(json_dir)

    # 1) map sample_id -> JSON (same convention the cutters used)
    json_paths = _sorted_jsons(json_dir)
    if len(json_paths) == 0:
        raise FileNotFoundError(f"No JSONs in {json_dir}")
    if sample_id < 1 or sample_id > len(json_paths):
        raise IndexError(f"sample_id {sample_id} out of range (found {len(json_paths)} jsons)")
    jpath = json_paths[sample_id - 1]

    # 2) load files for that sample
    uwb_file = sorted(uwb_dir.glob(f"*sample{sample_id}.npy"))[0]
    laser_file = sorted(laser_dir.glob(f"*sample{sample_id}.npy"))[0]
    uwb_patch = np.load(uwb_file)         # shape: (freq_bins, T)
    laser_seg = np.load(laser_file)

    # 3) reduce UWB spectrogram to 1-D over frequency
    if callable(reduce_fn):
        uwb_1d = reduce_fn(uwb_patch)
    elif reduce_fn == "mean":
        uwb_1d = uwb_patch.mean(axis=0)
    else:
        uwb_1d = uwb_patch.sum(axis=0)

    # ensure laser is 1-D
    if laser_seg.ndim > 1:
        # pick the axis that leaves time length largest
        laser_1d = laser_seg.mean(axis=0 if laser_seg.shape[0] > laser_seg.shape[1] else 1)
    else:
        laser_1d = laser_seg

    # 4) time from JSON duration
    with open(jpath, "r") as f:
        times = json.load(f)
    dur = _duration_seconds(times["start_dtime"], times["end_dtime"])  # seconds

    T_uwb = len(uwb_1d)
    T_las = len(laser_1d)
    if dur <= 0 or T_uwb < 2 or T_las < 2:
        raise ValueError("Bad duration or too few points.")

    t_uwb = np.linspace(0.0, dur, T_uwb, endpoint=False)
    t_las = np.linspace(0.0, dur, T_las, endpoint=False)

    # 5) put both on a common grid (choose the denser one)
    fs_uwb = T_uwb / dur
    fs_las = T_las / dur
    fs_common = max(fs_uwb, fs_las)
    N = int(round(dur * fs_common))
    t_common = np.linspace(0.0, dur, N, endpoint=False)

    uwb_rs = _lin_resample(uwb_1d, t_uwb, t_common)
    las_rs = _lin_resample(laser_1d, t_las, t_common)

    lag_ms, lag_samp = 0.0, 0
    if refine and N >= 4:
        lag_ms, lag_samp = _xcorr_lag_ms(uwb_rs, las_rs, fs_common)
        # shift UWB so it lines up with laser
        if lag_samp != 0:
            uwb_rs = np.roll(uwb_rs, -lag_samp)

    # 6) plot
    plt.figure(figsize=(12,4))
    plt.plot(t_common, _norm(uwb_rs), label="UWB (norm)")
    plt.plot(t_common, _norm(las_rs), label="Laser (norm)", alpha=0.85)
    plt.title(f"Aligned via Kinect JSON | lag≈{lag_ms:.1f} ms | dur={dur:.3f}s")
    plt.xlabel("Time (s)")
    plt.legend(); plt.tight_layout(); plt.show()

    print(f"UWB frames: {T_uwb} (~{fs_uwb:.1f} Hz) | Laser frames: {T_las} (~{fs_las:.1f} Hz)")
    print(f"JSON: {jpath.name}")
    print(f"Files:\n  UWB  -> {uwb_file.name}\n  Laser-> {laser_file.name}")

    return {
        "time": t_common, "uwb": uwb_rs, "laser": las_rs,
        "lag_ms": lag_ms, "duration": dur,
        "uwb_file": str(uwb_file), "laser_file": str(laser_file),
        "json_file": str(jpath),
    }

# --------- EDIT THESE to your real folders ----------
UWB_DIR   = "/ABSOLUTE/PATH/TO/.../UWB_processed/1/sentences5"
LASER_DIR = "/ABSOLUTE/PATH/TO/.../laser_processed/1/sentences5"
JSON_DIR  = "/ABSOLUTE/PATH/TO/.../<EXPID>_kinect_uwb/<session>/timestamps"  # must contain .json files
SAMPLE_ID = 5

if __name__ == "__main__":
    res = align_uwb_laser_by_json_no_scipy(
        UWB_DIR, LASER_DIR, JSON_DIR, SAMPLE_ID, reduce_fn="sum", refine=True
    )
