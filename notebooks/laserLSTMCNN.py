"""
laserLSTMCNN.py
================
Laser Signal 1D CNN + BiLSTM Encoder for Silent Speech Decoding
===============================================================

WHAT THIS FILE DOES
-------------------
1. Scans the laser_processed dataset directory and builds a list of all samples.
   Each sample = one .npy file = one raw 1D laser signal, labeled by group_name
   (e.g. "sentences1", "vowel3", "word7").
2. Trains a 1D CNN + Bidirectional LSTM that takes raw laser signals as input and
   outputs a 128-dim embedding + a classification prediction.
3. After training, runs inference on every sample and saves embeddings to an NPZ
   file — this is what you feed into the fusion layer later.

ARCHITECTURE DECISIONS
-----------------------
- ONE SAMPLE = ONE .npy file = one utterance recording by one user.
  Multiple files per group/user are treated as separate samples (free data).

- INPUT: raw 1D laser signal of shape (T,), T ~ 2600–9800 samples.
  Each signal is z-score normalized per-sample before processing.

- WHY CNN NOT STFT: the CNN learns its own temporal feature extraction from raw
  samples, which empirically outperforms hand-crafted STFT on this data. Without
  knowing the laser sensor's sampling rate we can't tune STFT window parameters,
  whereas the CNN adapts its receptive field to the actual signal structure.

- CNN: 3 strided Conv1d layers reduce temporal dim by 4×3×3 = 36×, producing a
  sequence of 128-dim feature vectors of length T' ≈ T/36 (~74–273 steps).
  Padding = kernel//2 so output lengths are easy to compute exactly.

- BILSTM: 2 layers, hidden=128 per direction → 256-dim final hidden state
  (concat of forward + backward last-layer hidden states).
  pack_padded_sequence is used with the CNN output lengths so padding frames
  are never processed by the LSTM recurrence.

- PROJECTION: Linear(256 → 128) + LayerNorm → raw 128-dim embedding.

- CLASSIFICATION HEAD: Linear(128 → num_classes), used during training only.
  Crucially, classification operates on the RAW embedding (not L2-normalized)
  so gradients flow cleanly through the full network.

- L2 NORMALIZATION: applied ONLY to the returned fusion embedding, not before
  the classifier. Normalizing before the classifier kills gradient magnitude.

- SPLIT: by user (users 17–18 val, 19–20 test) or random (USE_RANDOM_SPLIT=True).
  Use random split first to confirm the model can learn, then switch to
  user-based for speaker-independent evaluation.

OUTPUTS
-------
- laser_cnn_lstm_model.pt  : trained model weights
- laser_embeddings.npz     : embeddings for every sample, ready for fusion
  Keys: embeddings (N,128), labels (N,), user_ids (N,),
        group_names (N,), sample_names (N,)
- laser_label_map.json     : mapping from class index → label string
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

ROOT = "src/data/RVTALL/Processed_cut_data/laser_processed/"

# CNN architecture: 3 layers reduce temporal dim by 4×3×3 = 36×
# Padding = kernel//2 so output length ≈ input_length / stride
CNN_CHANNELS = [32, 64, 128]   # output channels per layer
CNN_KERNELS  = [15,  9,   7]   # kernel sizes
CNN_STRIDES  = [ 4,  3,   3]   # strides

# LSTM
HIDDEN_SIZE   = 128   # per direction; 256 total after bidirectional concat
NUM_LAYERS    = 2
EMBEDDING_DIM = 128
DROPOUT       = 0.3

# Training
BATCH_SIZE    = 32
LR            = 3e-4
EPOCHS        = 60
PATIENCE      = 20

# Split
VAL_USERS        = ["17", "18"]
TEST_USERS       = ["19", "20"]
USE_RANDOM_SPLIT = True   # True = random 75/15/10 (debug); False = user-based

# Output paths
MODEL_PATH      = "laser_cnn_lstm_model.pt"
EMBEDDINGS_PATH = "laser_embeddings.npz"
LABEL_MAP_PATH  = "laser_label_map.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# DATA LOADING
# ============================================================

def load_laser_signal(path):
    """
    Load one .npy laser file and z-score normalize it.
    Returns a float32 1D array of shape (T,).
    """
    sig = np.load(path).astype(np.float32).flatten()
    std = sig.std()
    if std > 1e-8:
        sig = (sig - sig.mean()) / std
    else:
        sig = sig - sig.mean()
    return sig


def build_sample_list(root):
    """
    Walk laser_processed and collect all valid sample paths.
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

class LaserDataset(Dataset):
    """
    Each item returns:
        sig    : (T,) float32 tensor — z-score normalized laser signal
        length : int scalar
        label  : int class index
        meta   : dict for bookkeeping
    """
    def __init__(self, samples, label_map, augment=False):
        self.augment = augment
        self.items   = []   # (sig_array, label_int, meta_dict)

        skipped = 0
        for s in samples:
            sig = load_laser_signal(s["path"])
            if len(sig) < 100:
                skipped += 1
                continue
            label = label_map[s["label_str"]]
            meta  = {k: s[k] for k in
                     ["user_id", "group_name", "sample_name", "path", "label_str"]}
            self.items.append((sig, label, meta))

        print(f"  Loaded {len(self.items)} samples ({skipped} skipped).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        sig, label, meta = self.items[idx]

        if self.augment:
            T = len(sig)
            # 1. Random temporal crop: keep 80–100% of the signal
            if T > 200:
                keep  = int(np.random.uniform(0.8, 1.0) * T)
                start = np.random.randint(0, T - keep + 1)
                sig   = sig[start:start + keep]
            # 2. Small additive Gaussian noise
            sig = sig + np.random.normal(0, 0.01, sig.shape).astype(np.float32)
            # 3. Random amplitude scaling
            sig = sig * float(np.random.uniform(0.9, 1.1))

        return torch.from_numpy(sig.astype(np.float32)), label, meta


def collate_fn(batch):
    """Pad variable-length 1D signals in a batch."""
    sigs, labels, metas = zip(*batch)
    lengths = torch.tensor([len(s) for s in sigs], dtype=torch.long)
    padded  = pad_sequence(sigs, batch_first=True)   # (B, T_max)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels, list(metas)


# ============================================================
# MODEL
# ============================================================

class LaserCNNLSTMEncoder(nn.Module):
    """
    1D CNN + BiLSTM encoder for raw laser signals → 128-dim embedding.

    Forward returns (logits, embedding_normed):
      - logits          : (B, num_classes) — used for training loss
      - embedding_normed: (B, 128) L2-normalized — used for fusion

    Classification is done on the RAW (pre-norm) embedding so gradients
    flow cleanly through the full network.
    """

    def __init__(self, num_classes,
                 cnn_channels=None, cnn_kernels=None, cnn_strides=None,
                 hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
                 embedding_dim=EMBEDDING_DIM, dropout=DROPOUT):
        super().__init__()

        cnn_channels = cnn_channels or CNN_CHANNELS
        cnn_kernels  = cnn_kernels  or CNN_KERNELS
        cnn_strides  = cnn_strides  or CNN_STRIDES

        # Store for output-length computation
        self._cnn_kernels  = cnn_kernels
        self._cnn_strides  = cnn_strides
        self._cnn_paddings = [k // 2 for k in cnn_kernels]

        # 1D CNN: Conv → BatchNorm → ReLU, stacked
        layers, in_ch = [], 1
        for out_ch, k, s, p in zip(cnn_channels, cnn_kernels,
                                   cnn_strides, self._cnn_paddings):
            layers += [
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(inplace=True),
            ]
            in_ch = out_ch
        self.cnn = nn.Sequential(*layers)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],   # 128
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # 256 = hidden_size * 2 directions
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _cnn_out_lengths(self, lengths):
        """Compute sequence lengths after all Conv1d layers."""
        L = lengths.float()
        for k, s, p in zip(self._cnn_kernels, self._cnn_strides, self._cnn_paddings):
            L = torch.floor((L + 2 * p - k) / s + 1)
        return L.long().clamp(min=1)

    def forward(self, x, lengths):
        """
        x       : (B, T_max) padded raw signals
        lengths : (B,) actual signal lengths before padding
        """
        # CNN feature extraction
        x = x.unsqueeze(1)       # (B, 1, T_max)
        x = self.cnn(x)          # (B, 128, T')
        x = x.permute(0, 2, 1)  # (B, T', 128)

        cnn_lengths = self._cnn_out_lengths(lengths)

        # BiLSTM with packing to skip padding frames
        packed = pack_padded_sequence(
            x, cnn_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers*2, B, hidden_size) — take last layer, both dirs
        h_fwd = h_n[-2]                        # (B, hidden_size)
        h_bwd = h_n[-1]                        # (B, hidden_size)
        h = torch.cat([h_fwd, h_bwd], dim=1)  # (B, 256)

        h = self.dropout(h)
        embedding        = self.embed_proj(h)                      # (B, 128) raw
        logits           = self.classifier(embedding)              # classify on raw
        embedding_normed = F.normalize(embedding, p=2, dim=1)     # for fusion only
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
        sig = load_laser_signal(s["path"])
        if len(sig) < 100:
            skipped += 1
            continue
        x      = torch.from_numpy(sig).unsqueeze(0).to(DEVICE)           # (1, T)
        length = torch.tensor([len(sig)], dtype=torch.long).to(DEVICE)
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
    print("\nLoading training data...")
    train_ds = LaserDataset(train_samples, label_map, augment=True)
    print("Loading validation data...")
    val_ds   = LaserDataset(val_samples,   label_map, augment=False)
    print("Loading test data...")
    test_ds  = LaserDataset(test_samples,  label_map, augment=False)

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
    model = LaserCNNLSTMEncoder(num_classes=num_classes).to(DEVICE)

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

Final validation accuracy: 0.301
accuracy | f1-score: 0.30, support: 763
macro avg | Precision: 0.29, Recall: 0.30, f1-score: 0.28, support: 763
weighted avg | Precision: 0.30, Recall: 0.30, f1-score: 0.29, support: 763

Test accuracy (unseen users ['19', '20']): 0.290
accuracy | f1-score: 0.29, support: 510
macro avg | Precision: 0.29, Recall: 0.29, f1-score: 0.27, support: 510
weighted avg | Precision: 0.31, Recall: 0.29, f1-score: 0.28, support: 510
"""