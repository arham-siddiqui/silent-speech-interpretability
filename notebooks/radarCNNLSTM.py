"""
radarCNNLSTM.py
================
Radar Range-Time Matrix (RTM) CNN + BiLSTM Encoder for Silent Speech Decoding
==============================================================================

WHAT THIS FILE DOES
-------------------
1. Scans the radar_processed dataset directory and builds a list of all samples.
   Each sample = one .npy file = a 2D range-time matrix (RTM), labeled by
   group_name (e.g. "sentences1", "vowel3", "word7").
2. Trains a 2D CNN + BiLSTM that takes the RTM as input and outputs a 128-dim
   embedding + a classification prediction.
3. After training, runs inference on every sample and saves embeddings to an NPZ
   file for use in the fusion layer.

DATA FORMAT
-----------
Each .npy file has shape (513, T) where:
  - 513 = fixed number of range bins (FFT of 1024-point CIR, so 512+1 bins)
  - T   = variable slow-time dimension (~150–552 steps, mean ~287)
Values are positive magnitudes (power spectrum), with extreme dynamic range
(5 to ~2.35M), so log-compression is applied before any model processing.
User 11 is absent from the dataset; all other users 1–20 are present.

ARCHITECTURE DECISIONS
-----------------------
- ONE SAMPLE = ONE RTM = one utterance recording by one user.
  Multiple samples per group/user are treated as separate samples (free data).

- PREPROCESSING:
  1. log1p(RTM) — critical: compresses 5→2.35M into a ~1–15 range
  2. Per-sample z-score normalization — removes per-user amplitude bias

- 2D CNN (range compression only):
  Four Conv2d layers with stride=(2,1) compress range 513→33 bins while
  preserving the full time axis T. Stride is applied ONLY along the range
  dimension (height) not along time (width), so temporal resolution is kept
  intact for the LSTM.

  513 → (stride 2) → 257 → 129 → 65 → 33 → mean-pool → 1
  Result after CNN: (B, 128, T) — 128-dim feature vector per time step.

- BILSTM: 2 layers, hidden=128 per direction → 256-dim final hidden state.
  pack_padded_sequence is used with original T lengths, so padded time steps
  are never processed by the LSTM recurrence.

- PROJECTION: Linear(256 → 128) + LayerNorm → raw 128-dim embedding.

- CLASSIFICATION HEAD: Linear(128 → num_classes), training only.
  Classification operates on the RAW embedding (not L2-normalized) so
  gradients flow cleanly through the full network.

- L2 NORMALIZATION: applied ONLY to the returned fusion embedding, not before
  the classifier.

- SPLIT: user-based (users 17–18 val, 19–20 test) or random (USE_RANDOM_SPLIT).
  Use random split first to confirm the model can learn, then switch to
  user-based for speaker-independent evaluation.

- DATA IS LOADED ON-THE-FLY: RTMs are large (~575KB each × ~5K files ≈ 2.8GB),
  so data is loaded in __getitem__ rather than preloaded in __init__.

OUTPUTS
-------
- radar_cnn_lstm_model.pt  : trained model weights
- radar_embeddings.npz     : embeddings for every sample, ready for fusion
  Keys: embeddings (N,128), labels (N,), user_ids (N,),
        group_names (N,), sample_names (N,)
- radar_label_map.json     : mapping from class index → label string
"""

import os
import re
import glob
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import classification_report

# ============================================================
# CONFIG — edit these to match your setup
# ============================================================

ROOT = "src/data/RVTALL/Processed_cut_data/radar_processed/"

# Fixed range dimension of the RTM
N_RANGE_BINS = 513

# 2D CNN: first 3 layers stride=(2,2) in both range+time; last layer stride=(2,1)
# Range: 513 → 257 → 129 → 65 → 33 → max-pool → 1
# Time:  T   → T/2 → T/4 → T/8 → T/8  (~36 avg steps into LSTM vs 287 before)
CNN_CHANNELS = [16, 32, 64, 128]   # output channels per layer

# LSTM
HIDDEN_SIZE   = 128   # per direction; 256 total after bidirectional concat
NUM_LAYERS    = 2
EMBEDDING_DIM = 128
DROPOUT       = 0.3

# Training
BATCH_SIZE    = 16    # smaller than laser/lip because RTMs are wider
LR            = 3e-4
EPOCHS        = 60
PATIENCE      = 20

# Split
VAL_USERS        = ["17", "18"]
TEST_USERS       = ["19", "20"]
USE_RANDOM_SPLIT = True   # True = random 75/15/10 (debug); False = user-based

# Output paths
MODEL_PATH      = "radar_cnn_lstm_model.pt"
EMBEDDINGS_PATH = "radar_embeddings.npz"
LABEL_MAP_PATH  = "radar_label_map.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# DATA LOADING
# ============================================================

def load_radar_rtm(path):
    """
    Load one .npy radar file and return a preprocessed (T, 513) float32 array.

    Steps:
      1. Load raw (513, T) matrix
      2. log1p compression — handles the 5→2.35M dynamic range
      3. Detrend: subtract each range bin's temporal mean to remove the static
         reflectivity component (which is ~0.93+ correlated across all classes
         and therefore carries zero discriminative information)
      4. Per-sample z-score normalization
      5. Transpose to (T, 513) so time is dim-0 for pad_sequence

    Returns float32 array of shape (T, 513), or None if T < 10.
    """
    rtm = np.load(path).astype(np.float32)   # (513, T)

    # Log compression
    rtm = np.log1p(rtm)

    # Detrend: remove each bin's temporal mean (static reflectivity = not discriminative)
    rtm = rtm - rtm.mean(axis=1, keepdims=True)

    # Per-sample z-score
    std = rtm.std()
    if std > 1e-8:
        rtm = rtm / std

    rtm = rtm.T   # (T, 513)

    if len(rtm) < 10:
        return None
    return rtm


def build_sample_list(root):
    """
    Walk radar_processed and collect all valid sample paths.
    Returns list of dicts: user_id, group_name, sample_name, path, label_str.
    """
    samples = []

    user_dirs = sorted(
        [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)],
        key=lambda p: (
            int(re.findall(r"\d+", os.path.basename(p))[0])
            if re.findall(r"\d+", os.path.basename(p)) else os.path.basename(p)
        )
    )

    if not user_dirs:
        raise RuntimeError(
            f"No user directories found under: {os.path.abspath(root)}\n"
            "Check your ROOT path."
        )

    print(f"Found {len(user_dirs)} user directories.")

    for user_dir in user_dirs:
        user_id = os.path.basename(user_dir)
        for group_prefix in ["sentences", "vowel", "word"]:
            for group_dir in sorted(glob.glob(os.path.join(user_dir, f"{group_prefix}*"))):
                group_name = os.path.basename(group_dir)
                for npy_path in sorted(glob.glob(os.path.join(group_dir, "*.npy"))):
                    samples.append({
                        "user_id":     user_id,
                        "group_name":  group_name,
                        "sample_name": os.path.basename(npy_path),
                        "path":        npy_path,
                        "label_str":   group_name,
                    })

    print(f"Total candidate samples: {len(samples)}")
    return samples


def build_label_map(samples):
    """Build sorted label → index mapping."""
    unique_labels = sorted(set(s["label_str"] for s in samples))
    return {lbl: idx for idx, lbl in enumerate(unique_labels)}


# ============================================================
# DATASET
# ============================================================

class RadarDataset(Dataset):
    """
    Each item returns:
        rtm    : (T, 513) float32 tensor — log-compressed, z-normed RTM
        length : int — number of time steps T
        label  : int class index
        meta   : dict for bookkeeping

    Data is loaded on-the-fly in __getitem__ (RTMs are large; preloading
    all ~5K files would require ~2.8GB of RAM).
    """
    def __init__(self, samples, label_map, augment=False):
        self.augment   = augment
        self.label_map = label_map
        self.items     = []   # (path, label_int, meta_dict) — no data preloaded

        skipped = 0
        for s in samples:
            # Validate the file exists and is loadable (quick shape check)
            try:
                shape = np.load(s["path"], mmap_mode="r").shape
                if len(shape) != 2 or shape[0] != N_RANGE_BINS or shape[1] < 10:
                    skipped += 1
                    continue
            except Exception:
                skipped += 1
                continue

            label = label_map[s["label_str"]]
            meta  = {k: s[k] for k in
                     ["user_id", "group_name", "sample_name", "path", "label_str"]}
            self.items.append((s["path"], label, meta))

        print(f"  Registered {len(self.items)} samples ({skipped} skipped).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, meta = self.items[idx]
        rtm = load_radar_rtm(path)   # (T, 513)

        if self.augment:
            T = len(rtm)
            # 1. Random temporal crop: keep 80–100% of time steps
            if T > 20:
                keep  = int(np.random.uniform(0.8, 1.0) * T)
                start = np.random.randint(0, T - keep + 1)
                rtm   = rtm[start:start + keep]

            # 2. Small additive noise on log-compressed values
            rtm = rtm + np.random.normal(0, 0.02, rtm.shape).astype(np.float32)

            # 3. Random range masking: zero a contiguous block of range bins
            if np.random.rand() < 0.5:
                r_start = np.random.randint(0, N_RANGE_BINS)
                r_width = np.random.randint(1, max(2, N_RANGE_BINS // 20))
                rtm[:, r_start:r_start + r_width] = 0.0

        return torch.from_numpy(rtm.astype(np.float32)), label, meta


def collate_fn(batch):
    """
    Pad variable-length (T, 513) RTMs along the time axis.
    Returns padded tensor of shape (B, T_max, 513).
    """
    rtms, labels, metas = zip(*batch)
    lengths = torch.tensor([len(r) for r in rtms], dtype=torch.long)
    padded  = pad_sequence(rtms, batch_first=True)   # (B, T_max, 513)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels, list(metas)


# ============================================================
# MODEL
# ============================================================

class RadarCNNLSTMEncoder(nn.Module):
    """
    2D CNN (range compression) + BiLSTM (temporal) encoder for radar RTMs.

    Architecture:
        Input (B, T_max, 513)
          → permute + unsqueeze → (B, 1, 513, T_max)
          → Conv2d(stride=(2,2)) × 3 + BN + ReLU   range: 513→257→129→65, time: T→T/2→T/4→T/8
          → Conv2d(stride=(2,1)) × 1 + BN + ReLU   range: 65→33,          time: T/8 (no change)
          → max over range dim → (B, 128, T/8)  [max not mean — focuses on active bins]
          → permute → (B, T/8, 128)
          → BiLSTM (2 layers, hidden=128/dir) with pack_padded_sequence
          → last hidden concat → (B, 256)
          → Dropout
          → Linear(256→128) + LayerNorm   ← raw embedding
          → classifier(raw embedding)     ← training loss
          → L2 normalize                  ← fusion output

    Key design choices vs. original:
      - stride=(2,2) in the first 3 layers reduces LSTM input from ~287 to ~36
        steps on average, making temporal patterns much easier to learn.
      - max-pool over range (not mean) focuses on the most activated range bins
        rather than averaging signal with uninformative background bins.
    """

    # Time-axis strides per CNN layer (for length tracking)
    TIME_STRIDES = [2, 2, 2, 1]

    def __init__(self, num_classes,
                 cnn_channels=None,
                 hidden_size=HIDDEN_SIZE,
                 num_layers=NUM_LAYERS,
                 embedding_dim=EMBEDDING_DIM,
                 dropout=DROPOUT):
        super().__init__()

        cnn_channels = cnn_channels or CNN_CHANNELS

        # Layers 0-2: stride=(2,2) — compress both range and time
        # Layer 3:    stride=(2,1) — compress range only (keep remaining time)
        layers, in_ch = [], 1
        for i, out_ch in enumerate(cnn_channels):
            t_stride = self.TIME_STRIDES[i]
            layers += [
                nn.Conv2d(in_ch, out_ch,
                          kernel_size=(3, 3),
                          stride=(2, t_stride),
                          padding=(1, 1)),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)
        # After CNN: range 513→33, time T→T/8 (approx)

        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],   # 128
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _time_out_lengths(self, lengths):
        """Compute time-axis output lengths after all CNN layers."""
        L = lengths.float()
        for t_stride in self.TIME_STRIDES:
            if t_stride > 1:
                # floor((L + 2*1 - 3) / stride) + 1 = floor((L-1)/stride) + 1
                L = torch.floor((L - 1) / t_stride) + 1
        return L.long().clamp(min=1)

    def forward(self, x, lengths):
        """
        x       : (B, T_max, 513) padded RTMs
        lengths : (B,) actual time lengths before padding
        """
        # Rearrange for 2D CNN: (B, 1, 513, T_max)
        x = x.permute(0, 2, 1).unsqueeze(1)

        x = self.cnn(x)            # (B, 128, R'=33, T')
        x = x.amax(dim=2)          # max over range → (B, 128, T') [not mean]
        x = x.permute(0, 2, 1)    # (B, T', 128) for LSTM

        # BiLSTM — pass downsampled time lengths
        cnn_lengths = self._time_out_lengths(lengths)
        packed = pack_padded_sequence(
            x, cnn_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, B, hidden_size) — last layer, both dirs
        h_fwd = h_n[-2]                        # (B, hidden_size)
        h_bwd = h_n[-1]                        # (B, hidden_size)
        h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 256)

        h = self.dropout(h)
        embedding        = self.embed_proj(h)                  # (B, 128) raw
        logits           = self.classifier(embedding)          # classify on raw
        embedding_normed = F.normalize(embedding, p=2, dim=1) # for fusion only
        return logits, embedding_normed

    def encode(self, x, lengths):
        """Inference-only: returns just the L2-normalized embedding."""
        with torch.no_grad():
            _, emb = self.forward(x, lengths)
        return emb


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for padded, lengths, labels, _ in loader:
        padded, lengths, labels = (
            padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        )
        optimizer.zero_grad()
        logits, _ = model(padded, lengths)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct     += (logits.argmax(1) == labels).sum().item()
        total       += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []
    for padded, lengths, labels, _ in loader:
        padded, lengths, labels = (
            padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        )
        logits, _ = model(padded, lengths)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        total      += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train(model, train_loader, val_loader, label_map):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc     = 0.0
    patience_counter = 0
    idx_to_label     = {v: k for k, v in label_map.items()}

    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(
            model, val_loader, criterion
        )
        scheduler.step()

        print(
            f"Epoch {epoch:3d}/{EPOCHS} | "
            f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
            f"Val loss {val_loss:.4f} acc {val_acc:.3f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ New best val acc: {best_val_acc:.3f} — model saved.")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, final_val_acc, val_preds, val_labels = eval_epoch(model, val_loader, criterion)
    print(f"\nFinal validation accuracy: {final_val_acc:.3f}")
    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    print("\nClassification report (validation set):")
    print(classification_report(
        val_labels, val_preds, target_names=target_names, zero_division=0
    ))
    return model


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_all_embeddings(model, all_samples, label_map):
    """Run inference on every sample and collect 128-dim embeddings."""
    model.eval()
    embeddings, labels, user_ids, group_names, sample_names = [], [], [], [], []

    print("\nExtracting embeddings for all samples...")
    skipped = 0

    for s in all_samples:
        rtm = load_radar_rtm(s["path"])
        if rtm is None:
            skipped += 1
            continue
        x      = torch.from_numpy(rtm).unsqueeze(0).to(DEVICE)           # (1, T, 513)
        length = torch.tensor([len(rtm)], dtype=torch.long).to(DEVICE)
        emb    = model.encode(x, length).squeeze(0).cpu().numpy()         # (128,)

        embeddings.append(emb)
        labels.append(label_map[s["label_str"]])
        user_ids.append(s["user_id"])
        group_names.append(s["group_name"])
        sample_names.append(s["sample_name"])

    print(f"  Extracted {len(embeddings)} embeddings ({skipped} skipped).")
    return (
        np.stack(embeddings, axis=0),           # (N, 128)
        np.array(labels,       dtype=np.int32),
        np.array(user_ids),
        np.array(group_names),
        np.array(sample_names),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # 1. Build sample list and label map
    all_samples = build_sample_list(ROOT)
    if not all_samples:
        print("No samples found. Check your ROOT path.")
        return

    label_map   = build_label_map(all_samples)
    num_classes = len(label_map)
    print(f"\nLabel map ({num_classes} classes):")
    for lbl, idx in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {idx:3d} → {lbl}")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    # 2. Split
    if USE_RANDOM_SPLIT:
        rng     = np.random.RandomState(42)
        idx     = rng.permutation(len(all_samples)).tolist()
        n_train = int(0.75 * len(idx))
        n_val   = int(0.15 * len(idx))
        train_samples = [all_samples[i] for i in idx[:n_train]]
        val_samples   = [all_samples[i] for i in idx[n_train:n_train + n_val]]
        test_samples  = [all_samples[i] for i in idx[n_train + n_val:]]
        print("Using RANDOM 75/15/10 split (debug mode — set USE_RANDOM_SPLIT=False for speaker-independent eval).")
    else:
        train_samples = [s for s in all_samples
                         if s["user_id"] not in VAL_USERS + TEST_USERS]
        val_samples   = [s for s in all_samples if s["user_id"] in VAL_USERS]
        test_samples  = [s for s in all_samples if s["user_id"] in TEST_USERS]
        print("Using USER-BASED split (speaker-independent eval).")

    print(f"Split: {len(train_samples)} train | "
          f"{len(val_samples)} val | {len(test_samples)} test samples")

    # 3. Datasets and loaders
    print("\nValidating training data...")
    train_ds = RadarDataset(train_samples, label_map, augment=True)
    print("Validating validation data...")
    val_ds   = RadarDataset(val_samples,   label_map, augment=False)
    print("Validating test data...")
    test_ds  = RadarDataset(test_samples,  label_map, augment=False)

    if len(train_ds) == 0:
        print("Training dataset is empty. Cannot proceed.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # 4. Build model
    model = RadarCNNLSTMEncoder(num_classes=num_classes).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # 5. Train
    model = train(model, train_loader, val_loader, label_map)

    # 6. Test evaluation
    if len(test_ds) > 0:
        criterion = nn.CrossEntropyLoss()
        _, test_acc, test_preds, test_labels_list = eval_epoch(
            model, test_loader, criterion
        )
        idx_to_label = {v: k for k, v in label_map.items()}
        target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
        print(f"\nTest accuracy (unseen users {TEST_USERS}): {test_acc:.3f}")
        print(classification_report(
            test_labels_list, test_preds,
            target_names=target_names, zero_division=0
        ))

    # 7. Extract embeddings for ALL samples and save
    embs, labels_arr, user_ids, group_names, sample_names = \
        extract_all_embeddings(model, all_samples, label_map)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embs,          # (N, 128) — THIS is your fusion input
        labels=labels_arr,
        user_ids=user_ids,
        group_names=group_names,
        sample_names=sample_names,
    )

    print(f"\nEmbeddings saved to {EMBEDDINGS_PATH}")
    print(f"  Shape: {embs.shape}  (N samples × 128 embedding dim)")
    print(f"\nTo load in your fusion script:")
    print(f"  data = np.load('{EMBEDDINGS_PATH}', allow_pickle=True)")
    print(f"  X = data['embeddings']   # shape (N, 128)")
    print(f"  y = data['labels']       # class indices")


if __name__ == "__main__":
    main()

"""
model stats

Final validation accuracy: 0.187
accuracy | f1-score: 0.19, support: 743
macro avg | Precision: 0.21, Recall: 0.20, f1-score: 0.19, support: 743
weighted avg | Precision: 0.20, Recall: 0.19, f1-score: 0.18, support: 743

Test accuracy (unseen users ['19', '20']): 0.183
accuracy | f1-score: 0.18, support: 496
macro avg | Precision: 0.19, Recall: 0.19, f1-score: 0.18, support: 496
weighted avg | Precision: 0.19, Recall: 0.18, f1-score: 0.17, support: 496
"""