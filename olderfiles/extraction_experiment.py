import numpy as np

path = "features/audio/sub1_sent1_audio_proc_0__audio.npz"  # or whatever yours is called
data = np.load(path, allow_pickle=True)

features = data["features"]   # shape: (T, 484)
meta = data["meta"].item()    # convert 0-d object array → dict

print(features.shape)
print(meta)

import matplotlib.pyplot as plt
import numpy as np

rms_mean = features[:, 240]  # one value per window

times = np.arange(len(rms_mean)) * meta["hop_sec"]  # center or start times
#plot RMS mean across 0.5 second windows

plt.plot(times, rms_mean)
plt.xlabel("time (s)")
plt.ylabel("RMS mean")
plt.title("RMS energy per 0.5 s window")
plt.show()

#Visualize “mel energy over time” from the windowed features

mel_means = features[:, 0:80]  # shape (T_windows, 80)

plt.imshow(mel_means.T, aspect="auto", origin="lower",
           extent=[0, len(mel_means)*meta["hop_sec"], 0, 80])
plt.xlabel("time (s)")
plt.ylabel("mel bin")
plt.title("Mel means per window")
plt.colorbar(label="mean dB")
plt.show()

#COMPARE DIFFERENT SESSIONS

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# where your npz files are
feat_dir = Path("features/audio")

# list of sessions you want to compare
# e.g. sent1_audio_proc_0__audio.npz ... sent10_audio_proc_0__audio.npz
sessions = [
    f"sub{s}_sent1_audio_proc_0__audio.npz" for s in range(1, 11)
]

plt.figure()

for sess in sessions:
    path = feat_dir / sess
    if not path.exists():
        print(f"Skipping {path} (not found)")
        continue

    data = np.load(path, allow_pickle=True)
    windows = data["features"]     # shape: [T_windows, 2F]
    meta = data["meta"].item()

    # F = 242 in our extractor (80 mel + 80 Δ + 80 ΔΔ + 1 rms + 1 f0)
    # window vector = [mean(0..F-1), std(0..F-1)] -> rms mean is index 240
    rms_mean = windows[:, 240]  # one value per 0.5 s window

    hop_sec = meta["hop_sec"]   # 0.25 in our config
    times = np.arange(len(rms_mean)) * hop_sec

    label = sess.split("__")[0]  # e.g. "sent1_audio_proc_0"
    plt.plot(times, rms_mean, label=label)

plt.xlabel("Time (s)")
plt.ylabel("RMS energy (mean per window)")
plt.title("RMS energy over time for subjects 1-10")
plt.legend()
plt.tight_layout()
plt.show()

