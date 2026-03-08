import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from sklearn.decomposition import PCA
from scipy.spatial.distance import euclidean
from scipy.interpolate import splprep, splev


# ---------------------- Utility ----------------------

def normalize_landmarks(lm):
    centroid = lm.mean(axis=0)
    lm_centered = lm - centroid
    scale = np.max(np.linalg.norm(lm_centered, axis=1)) + 1e-8
    return lm_centered / scale


def lip_height_width(lip):
    # Using inner-lip points (62 upper, 66 lower) in Dlib 68 format
    upper = lip[2]   # landmark 50+12? Actually for inner lips use indexes in lip
    lower = lip[6]
    
    # Outer corners: 48 (left), 54 (right)
    left_corner = lip[0]
    right_corner = lip[6]
    
    height = euclidean(upper, lower)
    width = euclidean(left_corner, right_corner)
    return height, width


def lip_area_perimeter(lip):
    x = lip[:, 0]
    y = lip[:, 1]

    # Shoelace polygon area
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    # Perimeter
    perimeter = np.sum(np.linalg.norm(lip - np.roll(lip, 1, axis=0), axis=1))

    return area, perimeter


def curvature(lip):
    x, y = lip[:, 0], lip[:, 1]
    tck, _ = splprep([x, y], s=0, per=True)
    u = np.linspace(0, 1, len(x))

    dx, dy = splev(u, tck, der=1)
    d2x, d2y = splev(u, tck, der=2)

    return np.abs(dx * d2y - dy * d2x) / (dx**2 + dy**2)**1.5


def pca_contour(lip, n_components=2):
    p = PCA(n_components=n_components)
    return p.fit_transform(lip).flatten()


def compute_velocity_acceleration(seq):
    seq = np.array(seq)
    vel = np.gradient(seq, axis=0)
    acc = np.gradient(vel, axis=0)
    return vel, acc


# ---------------------- Main Extractor ----------------------

def extract_features_from_video(video_dir):
    files = sorted(glob.glob(os.path.join(video_dir, "*.npy")))
    
    frames = [np.load(f) for f in files]

    # Extract only lip region: Dlib indices 48–67
    lip_frames = [f[48:68] for f in frames]
    lip_frames = [normalize_landmarks(lm) for lm in lip_frames]

    vel, acc = compute_velocity_acceleration(lip_frames)

    features_all = []

    for i, lm in enumerate(lip_frames):
        height, width = lip_height_width(lm)
        area, perimeter = lip_area_perimeter(lm)
        curv = curvature(lm)
        pca_feat = pca_contour(lm)

        features_all.append({
            "frame": i,
            "landmarks": lm,
            "lip_height": height,
            "lip_width": width,
            "lip_area": area,
            "lip_perimeter": perimeter,
            "velocity": vel[i],
            "acceleration": acc[i],
            "curvature": curv,
            "pca": pca_feat,
        })

    return features_all

def plot_lip_landmarks(features, frame_idx=0):
    """Plot lip landmarks for a specific frame."""
    lm = features[frame_idx]["landmarks"]
    plt.figure(figsize=(4,4))
    plt.scatter(lm[:,0], lm[:,1])
    plt.gca().invert_yaxis()
    plt.title(f"Lip Landmarks - Frame {frame_idx}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()


def plot_height_width(features):
    heights = [f["lip_height"] for f in features]
    widths  = [f["lip_width"] for f in features]

    plt.figure()
    plt.plot(heights, label="Lip Height")
    plt.plot(widths, label="Lip Width")
    plt.title("Lip Height & Width Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Distance")
    plt.legend()
    plt.show()


def plot_area_perimeter(features):
    areas = [f["lip_area"] for f in features]
    perims = [f["lip_perimeter"] for f in features]

    plt.figure()
    plt.plot(areas, label="Lip Area")
    plt.plot(perims, label="Lip Perimeter")
    plt.title("Lip Area & Perimeter Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.legend()
    plt.show()


def plot_velocity(features):
    """Plot velocity magnitude per frame."""
    vel_mag = [
        np.mean(np.linalg.norm(f["velocity"], axis=1)) 
        for f in features
    ]

    plt.figure()
    plt.plot(vel_mag)
    plt.title("Average Lip Landmark Velocity Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Velocity Magnitude")
    plt.show()


def plot_acceleration(features):
    """Plot acceleration magnitude per frame."""
    acc_mag = [
        np.mean(np.linalg.norm(f["acceleration"], axis=1)) 
        for f in features
    ]

    plt.figure()
    plt.plot(acc_mag)
    plt.title("Average Lip Landmark Acceleration Over Time")
    plt.xlabel("Frame")
    plt.ylabel("Acceleration Magnitude")
    plt.show()


def plot_curvature(features, frame_idx=0):
    curv = features[frame_idx]["curvature"]

    plt.figure()
    plt.plot(curv)
    plt.title(f"Lip Contour Curvature - Frame {frame_idx}")
    plt.xlabel("Contour Index")
    plt.ylabel("Curvature")
    plt.show()


def plot_pca(features, frame_idx=0):
    p = features[frame_idx]["pca"].reshape(-1, 2)

    plt.figure()
    plt.scatter(p[:,0], p[:,1])
    plt.title(f"PCA of Lip Contour - Frame {frame_idx}")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.axis("equal")
    plt.show()

#change this to local RVTALL path
video_dir = "src/data/rvtall/processed_cut_data/kinect_processed/1/sentences1/videos/video_0/landmarkers_cv"

features = extract_features_from_video(video_dir)

#lip landmark geometry
plot_lip_landmarks(features, frame_idx=5)

#Lip height & width over time
plot_height_width(features)

#Lip area & perimeter
plot_area_perimeter(features)

#Velocity
plot_velocity(features)

#Acceleration
plot_acceleration(features)

#Curvature
plot_curvature(features, frame_idx=10)

#PCA shape representation
plot_pca(features, frame_idx=10)

