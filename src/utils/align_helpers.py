from pathlib import Path
import json, numpy as np
from scipy import signal
import matplotlib.pyplot as plt

def _sorted_jsons(json_dir: Path):
    # must match their glob order used during cutting
    return sorted(json_dir.glob("*.json"))

def _load_window_from_json(jpath: Path):
    with open(jpath, "r") as f:
        t = json.load(f)
    # absolute seconds since epoch are irrelevant here; we only need duration
    # if you want absolute time, parse with datetime and convert.
    return t["start_dtime"], t["end_dtime"]

def _duration_seconds(start_str, end_str, fmt="%Y-%m-%d %H:%M:%S.%f"):
    from datetime import datetime
    s = datetime.strptime(start_str, fmt)
    e = datetime.strptime(end_str, fmt)
    return (e - s).total_seconds()

def align_uwb_laser_by_json(
    uwb_dir: str,
    laser_dir: str,
    json_dir: str,
    sample_id: int,
    reduce_fn="mean",          # or "sum" for higher SNR, or a custom callable
    xcorr_refine=True,
):
    uwb_dir = Path(uwb_dir); laser_dir = Path(laser_dir); json_dir = Path(json_dir)

    # 1) Map sample_id -> JSON used during cutting
    json_paths = _sorted_jsons(json_dir)
    if sample_id < 1 or sample_id > len(json_paths):
        raise IndexError("sample_id out of range for available JSONs.")
    jpath = json_paths[sample_id - 1]

    # 2) Load UWB patch & Laser segment
    #    (Choose the matching files for your subject/sentence; here we just pick by sample id)
    uwb_file = sorted(uwb_dir.glob(f"*sample{sample_id}.npy"))[0]
    laser_file = sorted(laser_dir.glob(f"*sample{sample_id}.npy"))[0]
    uwb_patch = np.load(uwb_file)       # shape: (doppler_bins≈205, T_uwb)
    laser_seg = np.load(laser_file)     # 1-D or 2-D depending on your pipeline

    if laser_seg.ndim > 1:
        laser_1d = laser_seg.mean(axis=1 if laser_seg.shape[0] == uwb_patch.shape[1] else 0)
    else:
        laser_1d = laser_seg

    # 3) Build time axes from the Kinect window
    start_s, end_s = _load_window_from_json(jpath)
    dur = _duration_seconds(start_s, end_s)   # seconds
    T_uwb = uwb_patch.shape[1]
    T_las = laser_1d.shape[0]
    t_uwb = np.linspace(0.0, dur, T_uwb, endpoint=False)
    t_las = np.linspace(0.0, dur, T_las, endpoint=False)

    # 4) Reduce UWB spectrogram -> 1-D amplitude
    if callable(reduce_fn):
        uwb_1d = reduce_fn(uwb_patch)
    elif reduce_fn == "sum":
        uwb_1d = uwb_patch.sum(axis=0)
    else:
        uwb_1d = uwb_patch.mean(axis=0)

    # 5) Optional: resample to common grid & fine lag via xcorr
    fs_common = max(T_uwb/dur, T_las/dur)
    N = int(round(dur * fs_common))
    uwb_rs = signal.resample(uwb_1d, N)
    las_rs = signal.resample(laser_1d, N)
    t = np.linspace(0, dur, N, endpoint=False)

    lag_ms = 0.0
    if xcorr_refine:
        x = (uwb_rs - uwb_rs.mean()) / (uwb_rs.std() + 1e-9)
        y = (las_rs - las_rs.mean()) / (las_rs.std() + 1e-9)
        corr = signal.correlate(x, y, mode="full", method="fft")
        lags = signal.correlation_lags(len(x), len(y), mode="full")
        k = np.argmax(corr)
        lag_samples = lags[k]
        lag_ms = 1000.0 * lag_samples / fs_common
        # shift UWB to align with Laser (positive = UWB lags)
        if lag_samples > 0:
            uwb_rs = np.pad(uwb_rs, (0,0))  # explicit for clarity
            uwb_rs = np.roll(uwb_rs, -lag_samples)
        elif lag_samples < 0:
            uwb_rs = np.roll(uwb_rs, -lag_samples)

    # 6) Quick plot
    plt.figure(figsize=(12,4))
    plt.plot(t, (uwb_rs-uwb_rs.min())/(uwb_rs.ptp()+1e-9), label="UWB (norm)")
    plt.plot(t, (las_rs-las_rs.min())/(las_rs.ptp()+1e-9), label="Laser (norm)", alpha=0.8)
    plt.title(f"Aligned via Kinect JSON  |  lag≈{lag_ms:.1f} ms")
    plt.xlabel("Time (s)")
    plt.legend(); plt.tight_layout(); plt.show()

    return {
        "uwb": uwb_rs, "laser": las_rs, "time": t, "lag_ms": lag_ms,
        "duration": dur, "frames_uwb": T_uwb, "frames_laser": T_las,
        "uwb_file": str(uwb_file), "laser_file": str(laser_file), "json": str(jpath),
    }
