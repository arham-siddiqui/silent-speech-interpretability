import os
import re
import glob
import numpy as np
import pandas as pd
from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev

# ============================================================
# Config
# ============================================================

ROOT = "src/data/rvtall/processed_cut_data/kinect_processed/"
OUTPUT_CSV = "lip_landmark_encodings.csv"
OUTPUT_NPZ = "lip_landmark_encodings.npz"

# Dlib 68-point lip slice: 48..67 inclusive => 20 points
LIP_START = 48
LIP_END = 68

# ============================================================
# Utility
# ============================================================

def normalize_landmarks(lm):
    """
    Normalize a single frame of landmarks by:
    1) centering at centroid
    2) scaling by max distance from centroid
    """
    lm = np.asarray(lm, dtype=np.float32)
    centroid = lm.mean(axis=0)
    lm_centered = lm - centroid
    scale = np.max(np.linalg.norm(lm_centered, axis=1)) + 1e-8
    return lm_centered / scale


def compute_velocity_acceleration(seq):
    """
    seq: (T, 20, 2)
    returns vel, acc with same shape
    """
    seq = np.asarray(seq, dtype=np.float32)
    vel = np.gradient(seq, axis=0)
    acc = np.gradient(vel, axis=0)
    return vel, acc


def lip_height_width(lip):
    """
    lip is shape (20,2), corresponding to original dlib indices 48..67.

    Correct mapping after slicing [48:68]:
      original 48 -> slice idx 0   (left corner)
      original 54 -> slice idx 6   (right corner)
      original 62 -> slice idx 14  (upper inner lip)
      original 66 -> slice idx 18  (lower inner lip)
    """
    left_corner = lip[0]
    right_corner = lip[6]
    upper_inner = lip[14]
    lower_inner = lip[18]

    height = euclidean(upper_inner, lower_inner)
    width = euclidean(left_corner, right_corner)
    return height, width


def lip_area_perimeter(lip):
    x = lip[:, 0]
    y = lip[:, 1]

    # Polygon area (shoelace)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Polygon perimeter
    perimeter = np.sum(np.linalg.norm(lip - np.roll(lip, 1, axis=0), axis=1))
    return area, perimeter


def curvature(lip):
    """
    Returns curvature sampled along the closed lip contour.
    Output shape: (20,)
    """
    x, y = lip[:, 0], lip[:, 1]

    try:
        tck, _ = splprep([x, y], s=0, per=True)
        u = np.linspace(0, 1, len(x))
        dx, dy = splev(u, tck, der=1)
        d2x, d2y = splev(u, tck, der=2)

        denom = (dx**2 + dy**2)**1.5 + 1e-8
        curv = np.abs(dx * d2y - dy * d2x) / denom
        return np.asarray(curv, dtype=np.float32)
    except Exception:
        # fallback if spline fit fails
        return np.zeros(len(lip), dtype=np.float32)


def temporal_stats(x, prefix):
    """
    x: 1D array over time
    returns dict of summary stats
    """
    x = np.asarray(x, dtype=np.float32)

    if len(x) == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
            f"{prefix}_range": 0.0,
            f"{prefix}_start": 0.0,
            f"{prefix}_end": 0.0,
            f"{prefix}_delta": 0.0,
        }

    return {
        f"{prefix}_mean": float(np.mean(x)),
        f"{prefix}_std": float(np.std(x)),
        f"{prefix}_min": float(np.min(x)),
        f"{prefix}_max": float(np.max(x)),
        f"{prefix}_range": float(np.max(x) - np.min(x)),
        f"{prefix}_start": float(x[0]),
        f"{prefix}_end": float(x[-1]),
        f"{prefix}_delta": float(x[-1] - x[0]),
    }


def list_sorted_npy_files(video_dir):
    """
    Sorts files numerically if names contain numbers.
    """
    files = glob.glob(os.path.join(video_dir, "*.npy"))

    def numeric_key(path):
        name = os.path.basename(path)
        nums = re.findall(r"\d+", name)
        return [int(n) for n in nums] if nums else [name]

    return sorted(files, key=numeric_key)


# ============================================================
# Per-video processing
# ============================================================

def load_lip_sequence(landmarkers_dir):
    """
    Loads all frames from one landmarkers_cv folder.
    Returns shape (T, 20, 2)
    """
    files = list_sorted_npy_files(landmarkers_dir)
    frames = []

    for f in files:
        arr = np.load(f)

        # Expect full face landmarks with at least 68 points
        if arr.ndim != 2 or arr.shape[0] < 68 or arr.shape[1] < 2:
            continue

        lip = arr[LIP_START:LIP_END, :2]
        lip = normalize_landmarks(lip)
        frames.append(lip)

    if len(frames) == 0:
        return None

    return np.asarray(frames, dtype=np.float32)  # (T, 20, 2)


def encode_lip_video(landmarkers_dir):
    """
    Converts one video's lip landmarks into a fixed-length feature vector.
    Returns:
      feature_vector: (D,)
      feature_names: list[str]
    """
    lip_seq = load_lip_sequence(landmarkers_dir)
    if lip_seq is None or len(lip_seq) < 2:
        return None, None

    T = lip_seq.shape[0]

    vel, acc = compute_velocity_acceleration(lip_seq)

    # Scalar time series
    heights = []
    widths = []
    areas = []
    perimeters = []
    vel_mag = []
    acc_mag = []
    curv_mean = []
    curv_std = []
    curv_max = []

    for t in range(T):
        lip = lip_seq[t]

        h, w = lip_height_width(lip)
        a, p = lip_area_perimeter(lip)
        c = curvature(lip)

        heights.append(h)
        widths.append(w)
        areas.append(a)
        perimeters.append(p)

        vel_mag.append(np.mean(np.linalg.norm(vel[t], axis=1)))
        acc_mag.append(np.mean(np.linalg.norm(acc[t], axis=1)))

        curv_mean.append(np.mean(c))
        curv_std.append(np.std(c))
        curv_max.append(np.max(c))

    heights = np.asarray(heights, dtype=np.float32)
    widths = np.asarray(widths, dtype=np.float32)
    areas = np.asarray(areas, dtype=np.float32)
    perimeters = np.asarray(perimeters, dtype=np.float32)
    vel_mag = np.asarray(vel_mag, dtype=np.float32)
    acc_mag = np.asarray(acc_mag, dtype=np.float32)
    curv_mean = np.asarray(curv_mean, dtype=np.float32)
    curv_std = np.asarray(curv_std, dtype=np.float32)
    curv_max = np.asarray(curv_max, dtype=np.float32)

    # Flattened static/dynamic summary blocks
    mean_landmarks = lip_seq.mean(axis=0).flatten()      # 20*2 = 40
    std_landmarks = lip_seq.std(axis=0).flatten()        # 40
    mean_velocity = vel.mean(axis=0).flatten()           # 40
    std_velocity = vel.std(axis=0).flatten()             # 40
    mean_accel = acc.mean(axis=0).flatten()              # 40
    std_accel = acc.std(axis=0).flatten()                # 40

    feature_dict = {}

    # Landmark summary features
    for i, val in enumerate(mean_landmarks):
        feature_dict[f"mean_landmark_{i}"] = float(val)
    for i, val in enumerate(std_landmarks):
        feature_dict[f"std_landmark_{i}"] = float(val)
    for i, val in enumerate(mean_velocity):
        feature_dict[f"mean_velocity_{i}"] = float(val)
    for i, val in enumerate(std_velocity):
        feature_dict[f"std_velocity_{i}"] = float(val)
    for i, val in enumerate(mean_accel):
        feature_dict[f"mean_accel_{i}"] = float(val)
    for i, val in enumerate(std_accel):
        feature_dict[f"std_accel_{i}"] = float(val)

    # Temporal scalar summary features
    for d in [
        temporal_stats(heights, "lip_height"),
        temporal_stats(widths, "lip_width"),
        temporal_stats(areas, "lip_area"),
        temporal_stats(perimeters, "lip_perimeter"),
        temporal_stats(vel_mag, "vel_mag"),
        temporal_stats(acc_mag, "acc_mag"),
        temporal_stats(curv_mean, "curv_mean"),
        temporal_stats(curv_std, "curv_std"),
        temporal_stats(curv_max, "curv_max"),
    ]:
        feature_dict.update(d)

    # Sequence length can be useful too
    feature_dict["num_frames"] = float(T)

    feature_names = list(feature_dict.keys())
    feature_vector = np.array([feature_dict[k] for k in feature_names], dtype=np.float32)

    return feature_vector, feature_names


# ============================================================
# Dataset traversal
# ============================================================

def find_all_landmarkers_dirs(root):
    """
    Finds all landmarkers_cv directories under:
    user/{sentences*, vowels*, words*}/videos/video_*/landmarkers_cv
    """
    results = []

    print("ROOT:", root)
    print("ABS ROOT:", os.path.abspath(root))
    print("ROOT EXISTS:", os.path.exists(root))

    user_dirs = sorted([d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)])
    print("user_dirs:", user_dirs[:5], " ... total =", len(user_dirs))

    for user_dir in user_dirs:
        print("\nUSER DIR:", user_dir)

        for group_prefix in ["sentences", "vowels", "words"]:
            group_dirs = sorted(glob.glob(os.path.join(user_dir, f"{group_prefix}*")))
            print(f"  {group_prefix} matches:", [os.path.basename(g) for g in group_dirs])

            for group_dir in group_dirs:
                videos_dir = os.path.join(group_dir, "videos")
                print("    checking videos_dir:", videos_dir, "exists =", os.path.isdir(videos_dir))

                if not os.path.isdir(videos_dir):
                    continue

                video_dirs = sorted(glob.glob(os.path.join(videos_dir, "video_*")))
                print("    video dirs:", [os.path.basename(v) for v in video_dirs[:5]], "total =", len(video_dirs))

                for video_dir in video_dirs:
                    landmarkers_dir = os.path.join(video_dir, "landmarkers_cv")
                    print("      checking:", landmarkers_dir, "exists =", os.path.isdir(landmarkers_dir))

                    if os.path.isdir(landmarkers_dir):
                        results.append({
                            "user_id": os.path.basename(user_dir),
                            "group_type": group_prefix,
                            "group_name": os.path.basename(group_dir),
                            "video_name": os.path.basename(video_dir),
                            "landmarkers_dir": landmarkers_dir,
                        })

    return results


# ============================================================
# Main
# ============================================================

def main():
    entries = find_all_landmarkers_dirs(ROOT)
    print(f"Found {len(entries)} landmarkers_cv folders.")

    rows = []
    feature_matrix = []
    feature_names_master = None

    for idx, entry in enumerate(entries):
        landmarkers_dir = entry["landmarkers_dir"]
        print(f"[{idx+1}/{len(entries)}] Processing: {landmarkers_dir}")

        try:
            vec, feature_names = encode_lip_video(landmarkers_dir)

            if vec is None:
                print("  -> skipped (not enough valid frames)")
                continue

            if feature_names_master is None:
                feature_names_master = feature_names
            else:
                if feature_names != feature_names_master:
                    raise ValueError("Feature name mismatch across videos.")

            row = {
                "user_id": entry["user_id"],
                "group_type": entry["group_type"],
                "group_name": entry["group_name"],
                "video_name": entry["video_name"],
                "landmarkers_dir": entry["landmarkers_dir"],
            }

            for name, val in zip(feature_names, vec):
                row[name] = float(val)

            rows.append(row)
            feature_matrix.append(vec)

        except Exception as e:
            print(f"  -> error: {e}")

    if len(rows) == 0:
        print("No valid encodings generated.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False)

    X = np.stack(feature_matrix, axis=0)  # (N, D)
    metadata = df[["user_id", "group_type", "group_name", "video_name", "landmarkers_dir"]].to_dict(orient="records")

    np.savez_compressed(
        OUTPUT_NPZ,
        X=X,
        feature_names=np.array(feature_names_master, dtype=object),
        metadata=np.array(metadata, dtype=object),
    )

    print("\nDone.")
    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved NPZ: {OUTPUT_NPZ}")
    print(f"Num samples: {X.shape[0]}")
    print(f"Feature dim: {X.shape[1]}")


if __name__ == "__main__":
    main()