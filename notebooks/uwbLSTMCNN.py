"""
uwbLSTMCNN.py
=============
UWB Radar (7.5 GHz CIR) encoder for silent speech decoding.

WHAT THIS FILE DOES
-------------------
1. Loads UWB range-time matrices from uwb_processed/, pairing the two
   antennas per recording as a 2-channel input
2. Trains a 2D CNN + BiLSTM that compresses each (2, 205, T) RTM into
   a 128-dim L2-normalised embedding
3. Saves embeddings for all samples to uwb_embeddings.npz for the fusion layer

DATA STRUCTURE
--------------
uwb_processed/
└── {user_id}/                         # 1-20
    └── {sentences_1..10|vowel_1..5|word_1..15}/
        └── {user}_{type}_{1}_{antenna}_{sample}.npy   # antenna ∈ {1, 2}

Each .npy: (205, T) float64 — range-time matrix.
T varies across recordings (108-331, mean ~181).
Two antennas are ALWAYS both present per recording.
Antennas are stacked as a 2-channel input: (2, 205, T).

PREPROCESSING
-------------
- Stack antennas → (2, 205, T)
- Detrend: subtract per-range-bin temporal mean per antenna
  (removes static background reflectivity)
- Z-score normalise by global std
- Transpose → (T, 2, 205) so pad_sequence pads along T

ARCHITECTURE
------------
Input (B, 2, 205, T_max)
  → 4-layer 2D CNN
      channels: 2 → 32 → 64 → 128 → 128
      kernel (3,3), padding (1,1)
      stride (2,2) for layers 1-3; stride (2,1) for layer 4
      range bins: 205 → 103 → 52 → 26 → 13
      time: T → T/2 → T/4 → T/8 → T/8
  → max-pool over range → (B, 128, T/8)
  → BiLSTM (1 layer, hidden=128 per direction)
  → concat fwd+bwd → (B, 256)
  → Dropout
  → Linear(256 → 128) + LayerNorm
  → L2 normalise  ← EMBEDDING (fusion input)
  → Linear(128 → num_classes)  ← classification head, training only

LABEL NORMALISATION
-------------------
Folder names use underscores (sentences_1, vowel_1, word_1).
Normalised to (sentences1, vowel1, word1) to match other
modality NPZs and align correctly in the fusion script.

OUTPUTS
-------
- uwb_cnn_lstm_model.pt
- uwb_embeddings.npz
    embeddings   (N, 128)
    labels       (N,)
    user_ids     (N,)
    group_names  (N,)  ← normalised, e.g. "sentences1"
    sample_names (N,)
- uwb_label_map.json
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from sklearn.metrics import classification_report
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

ROOT = "src/data/RVTALL/Processed_cut_data/uwb_processed/"

# Only use users 1-20 to match other modalities
MAX_USER_ID = 20

HIDDEN_SIZE   = 128   # per direction; 256 after BiLSTM concat
NUM_LAYERS    = 1
EMBEDDING_DIM = 128
DROPOUT       = 0.3

BATCH_SIZE = 16        # (B, 2, 205, T_max) — smaller batch for memory
LR         = 3e-4
EPOCHS     = 60
PATIENCE   = 20

VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]

USE_RANDOM_SPLIT = False   # False = speaker-independent user-based split

MODEL_PATH      = "uwb_cnn_lstm_model.pt"
EMBEDDINGS_PATH = "uwb_embeddings.npz"
LABEL_MAP_PATH  = "uwb_label_map.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# DATA LOADING
# ============================================================

def normalise_group_name(folder_name: str) -> str:
    """
    sentences_1  → sentences1
    vowel_1      → vowel1
    word_1       → word1
    Removes the first underscore to match other modality label formats.
    """
    return folder_name.replace("_", "", 1)


EXPECTED_RANGE_BINS = 205   # a small number of files have 2048 range bins (corrupted)

def load_uwb_rtm(ant1_path: str, ant2_path: str):
    """
    Load both antenna files, stack as (2, 205, T), detrend, normalise.
    Returns (T, 2, 205) float32, or None if too short or wrong shape.
    """
    rtm1 = np.load(ant1_path).astype(np.float32)   # (205, T)
    rtm2 = np.load(ant2_path).astype(np.float32)   # (205, T)

    # Skip corrupted files with wrong range dimension (25 files have 2048 bins)
    if rtm1.shape[0] != EXPECTED_RANGE_BINS or rtm2.shape[0] != EXPECTED_RANGE_BINS:
        return None

    # Ensure consistent T (should always match, but guard just in case)
    T = min(rtm1.shape[1], rtm2.shape[1])
    rtm1, rtm2 = rtm1[:, :T], rtm2[:, :T]

    # Stack antennas as channels: (2, 205, T)
    rtm = np.stack([rtm1, rtm2], axis=0)

    # Detrend per antenna per range bin (remove static background)
    rtm = rtm - rtm.mean(axis=2, keepdims=True)

    # Global normalise
    std = rtm.std()
    if std > 1e-8:
        rtm = rtm / std

    # Transpose to (T, 2, 205) so pad_sequence can pad along T
    rtm = rtm.transpose(2, 0, 1)   # (T, 2, 205)

    if T < 10:
        return None
    return rtm


def build_sample_list(root: str):
    """
    Walk uwb_processed and collect all (antenna1, antenna2) file pairs.
    Returns list of dicts: user_id, group_name (normalised), sample_name,
                           ant1_path, ant2_path, label_str
    """
    samples = []

    user_dirs = sorted(
        [d for d in os.listdir(root)
         if os.path.isdir(os.path.join(root, d)) and d.isdigit()
         and int(d) <= MAX_USER_ID],
        key=lambda x: int(x)
    )

    if not user_dirs:
        raise RuntimeError(f"No user directories found under: {os.path.abspath(root)}")

    print(f"Found {len(user_dirs)} user directories.")

    for user in user_dirs:
        user_path = os.path.join(root, user)

        for grp_folder in sorted(os.listdir(user_path)):
            grp_path = os.path.join(user_path, grp_folder)
            if not os.path.isdir(grp_path):
                continue
            if not any(grp_folder.startswith(p) for p in
                       ["sentences_", "vowel_", "word_"]):
                continue

            label_str = normalise_group_name(grp_folder)  # e.g. sentences1

            # Group files by sample id, collect antenna paths
            by_sample = defaultdict(dict)
            for fname in os.listdir(grp_path):
                if not fname.endswith(".npy"):
                    continue
                parts = fname.replace(".npy", "").split("_")
                if len(parts) < 5:
                    continue
                ant       = parts[3]             # '1' or '2'
                sample_id = "_".join(parts[4:])  # 'sample1', 'sample10', ...
                by_sample[sample_id][ant] = os.path.join(grp_path, fname)

            for sample_id, ant_files in sorted(by_sample.items()):
                if "1" not in ant_files or "2" not in ant_files:
                    continue  # skip incomplete pairs (shouldn't happen)
                samples.append({
                    "user_id":     user,
                    "group_name":  label_str,
                    "sample_name": f"{grp_folder}_{sample_id}",
                    "ant1_path":   ant_files["1"],
                    "ant2_path":   ant_files["2"],
                    "label_str":   label_str,
                })

    print(f"Total candidate samples: {len(samples)}")
    return samples


def build_label_map(samples):
    unique = sorted(set(s["label_str"] for s in samples))
    return {lbl: idx for idx, lbl in enumerate(unique)}


# ============================================================
# DATASET
# ============================================================

class UWBDataset(Dataset):
    """
    Loads RTM pairs on-the-fly in __getitem__.
    Returns:
        rtm    : (T, 2, 205) tensor
        label  : int
        meta   : dict
    """
    def __init__(self, samples, label_map, augment=False):
        self.label_map = label_map
        self.augment   = augment
        self.items     = list(samples)
        print(f"  Dataset ready: {len(self.items)} samples.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        s   = self.items[idx]
        rtm = load_uwb_rtm(s["ant1_path"], s["ant2_path"])

        # Fallback on load failure (very rare)
        if rtm is None:
            alt = self.items[(idx + 1) % len(self.items)]
            rtm = load_uwb_rtm(alt["ant1_path"], alt["ant2_path"])
        if rtm is None:
            rtm = np.zeros((20, 2, 205), dtype=np.float32)

        if self.augment:
            T = len(rtm)
            # Random temporal crop: keep 80-100% of frames
            if T > 10:
                keep  = int(np.random.uniform(0.8, 1.0) * T)
                start = np.random.randint(0, T - keep + 1)
                rtm   = rtm[start:start + keep]
            # Small Gaussian noise
            rtm = rtm + np.random.normal(0, 0.01, rtm.shape).astype(np.float32)
            # Random amplitude scaling
            rtm = rtm * float(np.random.uniform(0.9, 1.1))

        rtm_tensor = torch.from_numpy(rtm.astype(np.float32))  # (T, 2, 205)
        label      = self.label_map[s["label_str"]]
        return rtm_tensor, label, {k: s[k] for k in
               ["user_id", "group_name", "sample_name", "label_str"]}


def collate_fn(batch):
    rtms, labels, metas = zip(*batch)
    lengths = torch.tensor([len(r) for r in rtms], dtype=torch.long)
    padded  = pad_sequence(rtms, batch_first=True)   # (B, T_max, 2, 205)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels, list(metas)


# ============================================================
# MODEL
# ============================================================

class UWBEncoder(nn.Module):
    """
    2-channel (dual-antenna) 2D CNN over (range=205, time) followed by
    a BiLSTM over the downsampled time axis.

    Input:  (B, T_max, 2, 205)
    CNN:    reduces range 205→13, time T→T/8
    LSTM:   over T/8 time steps
    Output: 128-dim L2-normalised embedding + classification logits
    """

    # Time stride per CNN layer (range stride is always 2)
    TIME_STRIDES = [2, 2, 2, 1]

    def __init__(self, num_classes: int, hidden_size: int = 128,
                 num_layers: int = 1, embedding_dim: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        def cnn_block(in_c, out_c, t_stride):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=(3, 3),
                          stride=(2, t_stride), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_c),
                nn.GELU(),
            )

        self.cnn = nn.Sequential(
            cnn_block(2,   32,  self.TIME_STRIDES[0]),  # → (32, 103, T/2)
            cnn_block(32,  64,  self.TIME_STRIDES[1]),  # → (64,  52, T/4)
            cnn_block(64,  128, self.TIME_STRIDES[2]),  # → (128, 26, T/8)
            cnn_block(128, 128, self.TIME_STRIDES[3]),  # → (128, 13, T/8)
        )

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout    = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def _time_out_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        """Track time-axis length through CNN layers for pack_padded_sequence."""
        L = lengths.float()
        for t_stride in self.TIME_STRIDES:
            if t_stride > 1:
                # padding=1, kernel=3: floor((L + 2 - 3) / stride + 1)
                L = torch.floor((L + 2 - 3) / t_stride + 1)
        return L.long().clamp(min=1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x       : (B, T_max, 2, 205)
        lengths : (B,) actual T per sample
        Returns:
            logits    : (B, num_classes)
            embedding : (B, 128) L2-normalised
        """
        # Rearrange for Conv2d: (B, channels=2, range=205, time=T_max)
        x = x.permute(0, 2, 3, 1)   # (B, 2, 205, T_max)

        x = self.cnn(x)              # (B, 128, 13, T')

        # Max-pool over range (focuses on most activated range bins)
        x = x.amax(dim=2)            # (B, 128, T')

        x = x.permute(0, 2, 1)      # (B, T', 128) for LSTM

        cnn_lengths = self._time_out_lengths(lengths)
        packed = pack_padded_sequence(
            x, cnn_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)

        h_fwd = h_n[-2]
        h_bwd = h_n[-1]
        h = torch.cat([h_fwd, h_bwd], dim=1)   # (B, 256)

        h         = self.dropout(h)
        embedding = self.embed_proj(h)                         # (B, 128) raw
        logits    = self.classifier(embedding)                 # classify on raw
        embedding_normed = F.normalize(embedding, p=2, dim=1)  # L2 for fusion
        return logits, embedding_normed

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
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
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
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
        loss   = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds   = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_labels


def train(model, train_loader, val_loader, label_map):
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    idx_to_label = {v: k for k, v in label_map.items()}

    best_val_acc = 0.0
    patience_ctr = 0

    print("\n" + "=" * 60)
    print("TRAINING UWB ENCODER")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, val_preds, val_labels = eval_epoch(
            model, val_loader, criterion
        )
        scheduler.step()

        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.3f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ New best val acc: {best_val_acc:.3f} — model saved.")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, final_val_acc, val_preds, val_labels = eval_epoch(
        model, val_loader, criterion
    )
    print(f"\nFinal validation accuracy: {final_val_acc:.3f}")
    target_names = [idx_to_label[i] for i in range(len(idx_to_label))]
    print(classification_report(val_labels, val_preds,
                                target_names=target_names, zero_division=0))
    return model


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_all_embeddings(model, all_samples, label_map):
    model.eval()
    embeddings, labels, user_ids, group_names, sample_names = [], [], [], [], []
    skipped = 0

    print("\nExtracting embeddings for all samples...")
    for s in all_samples:
        rtm = load_uwb_rtm(s["ant1_path"], s["ant2_path"])
        if rtm is None:
            skipped += 1
            continue

        x      = torch.from_numpy(rtm.astype(np.float32)).unsqueeze(0).to(DEVICE)
        length = torch.tensor([len(rtm)], dtype=torch.long).to(DEVICE)

        emb = model.encode(x, length).squeeze(0).cpu().numpy()  # (128,)
        embeddings.append(emb)
        labels.append(label_map[s["label_str"]])
        user_ids.append(s["user_id"])
        group_names.append(s["group_name"])
        sample_names.append(s["sample_name"])

    print(f"  Extracted {len(embeddings)} embeddings ({skipped} skipped).")
    return (
        np.stack(embeddings),
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
        print("No samples found. Check ROOT path.")
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
        print("Using RANDOM 75/15/10 split (debug — set USE_RANDOM_SPLIT=False for final).")
    else:
        train_samples = [s for s in all_samples
                         if s["user_id"] not in VAL_USERS + TEST_USERS]
        val_samples   = [s for s in all_samples if s["user_id"] in VAL_USERS]
        test_samples  = [s for s in all_samples if s["user_id"] in TEST_USERS]
        print("Using USER-BASED split (speaker-independent).")

    print(f"Split: {len(train_samples)} train | "
          f"{len(val_samples)} val | {len(test_samples)} test")

    # 3. Datasets and loaders
    print("\nBuilding datasets (on-the-fly loading)...")
    train_ds = UWBDataset(train_samples, label_map, augment=True)
    val_ds   = UWBDataset(val_samples,   label_map, augment=False)
    test_ds  = UWBDataset(test_samples,  label_map, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # 4. Build model
    model = UWBEncoder(
        num_classes=num_classes,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    with torch.no_grad():
        dummy_T   = torch.tensor([200, 150, 108])
        cnn_T_out = model._time_out_lengths(dummy_T)
        print(f"CNN length check: input T={dummy_T.tolist()} → CNN output T={cnn_T_out.tolist()}")

    # 5. Train
    model = train(model, train_loader, val_loader, label_map)

    # 6. Test evaluation
    if len(test_ds) > 0:
        criterion = nn.CrossEntropyLoss()
        _, test_acc, test_preds, test_labels = eval_epoch(
            model, test_loader, criterion
        )
        split_label = (f"unseen users {TEST_USERS}" if not USE_RANDOM_SPLIT
                       else "random test split (NOT speaker-disjoint)")
        print(f"\nTest accuracy ({split_label}): {test_acc:.3f}")
        idx_to_label = {v: k for k, v in label_map.items()}
        print(classification_report(
            test_labels, test_preds,
            target_names=[idx_to_label[i] for i in range(num_classes)],
            zero_division=0,
        ))

    # 7. Extract and save embeddings for all samples
    embs, labels_arr, user_ids, group_names, sample_names = \
        extract_all_embeddings(model, all_samples, label_map)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embs,
        labels=labels_arr,
        user_ids=user_ids,
        group_names=group_names,
        sample_names=sample_names,
    )
    print(f"\nEmbeddings saved to {EMBEDDINGS_PATH}")
    print(f"  Shape: {embs.shape}   (N samples × 128 embedding dim)")
    print(f"\nTo load in fusion script:")
    print(f"  d = np.load('{EMBEDDINGS_PATH}', allow_pickle=True)")
    print(f"  X = d['embeddings']   # (N, 128)")
    print(f"  y = d['labels']")


if __name__ == "__main__":
    main()

"""
model stats

Final validation accuracy: 0.161
accuracy | f1-score: 0.16, support: 540
macro avg | Precision: 0.21, Recall: 0.17, f1-score: 0.16, support: 540
weighted avg | Precision: 0.21, Recall: 0.16, f1-score: 0.16, support: 540

Test accuracy (unseen users ['19', '20']): 0.224
accuracy | f1-score: 0.22, support: 540
macro avg | Precision: 0.25, Recall: 0.22, f1-score: 0.20, support: 540
weighted avg | Precision: 0.25, Recall: 0.22, f1-score: 0.20, support: 540
"""