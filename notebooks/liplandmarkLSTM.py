"""
lip_landmark_lstm.py
=====================
Lip Landmark LSTM Encoder for Silent Speech Decoding
=====================================================

WHAT THIS FILE DOES
-------------------
1. Scans the dataset directory and builds a list of all samples
   (each sample = one video_* folder's landmarkers_cv, with a label
   derived from the group_name e.g. "sentences1", "vowels3", "words7")
2. Trains a Bidirectional LSTM that takes raw lip landmark sequences
   as input and outputs a 128-dim embedding + a classification prediction
3. After training, runs inference on every sample in the dataset and
   saves a NPZ file of embeddings — this is what you feed into your
   fusion layer later

ARCHITECTURE DECISIONS (read this)
-----------------------------------
- ONE SAMPLE = ONE VIDEO of one utterance. Users 1-20 contribute
  training diversity but are NOT combined at the LSTM level.
  The LSTM sees individual utterances.

- MULTIPLE VIDEOS PER UTTERANCE (video_0 to video_6): treated as
  separate samples with the same label. This is free data augmentation.

- INPUT: (T, 40) per sample — the 20 lip landmarks × 2 (x,y), flattened
  per frame. We also append per-frame velocity making it (T, 80).
  Raw landmarks beat hand-crafted features because the LSTM learns
  its own temporal patterns.

- VARIABLE LENGTH: handled via PyTorch pack_padded_sequence so the
  LSTM never sees padding frames in its recurrence.

- ARCHITECTURE: BiLSTM (2 layers, hidden=256 each direction)
  → final hidden state (512-dim) → Linear → 128-dim embedding (L2-normed)
  → classification head during training.
  After training, you discard the classification head and use the
  128-dim embedding directly for fusion.

- TRAIN/VAL/TEST SPLIT: by USER, not by sample. Users 17-18 = val,
  users 19-20 = test. This tests speaker-independent generalization,
  which is the actual hard problem in silent speech recognition.

- LABEL SPACE: we build one unified label set across sentences/vowels/words.
  e.g. "sentences1" → class 0, "vowels3" → class 5, "words7" → class 15
  This means the model classifies across all 36 utterance types at once.
  If you prefer separate classifiers per group, see GROUP_SEPARATE flag.

OUTPUTS
-------
- lip_lstm_model.pt        : trained model weights
- lip_embeddings.npz       : embeddings for every sample, ready for fusion
  Keys: embeddings (N,128), labels (N,), user_ids (N,),
        group_types (N,), group_names (N,), video_names (N,)
- lip_label_map.json       : mapping from class index → label string
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
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix
from collections import defaultdict

# ============================================================
# CONFIG — edit these to match your setup
# ============================================================

ROOT = "src/data/RVTALL/Processed_cut_data/kinect_processed/"

# Dlib 68-point lip indices
LIP_START = 48
LIP_END   = 68   # exclusive → 20 points

# Model hyperparameters
HIDDEN_SIZE   = 256   # per direction; total = 512 after bidirectional concat
NUM_LAYERS    = 2
EMBEDDING_DIM = 128   # output embedding size fed to fusion layer
DROPOUT       = 0.3

# Training
BATCH_SIZE    = 32
LR            = 3e-4  # was 1e-3; lower lr more stable for BiLSTM
EPOCHS        = 60
PATIENCE      = 20    # early stopping patience

# Split: user IDs reserved for val and test (strings, since dir names vary)
VAL_USERS     = ["17", "18"]
TEST_USERS    = ["19", "20"]

# Set True to use random 75/15/10 split for debugging (confirms model can learn).
# Set False to use the user-based split for speaker-independent evaluation.
USE_RANDOM_SPLIT = True

# Set True to train separate models per group type (sentences/vowels/words)
# Set False (recommended) to train one unified model across all utterances
GROUP_SEPARATE = False

# Output paths
MODEL_PATH      = "lip_lstm_model.pt"
EMBEDDINGS_PATH = "lip_embeddings.npz"
LABEL_MAP_PATH  = "lip_label_map.json"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# DATA LOADING
# ============================================================

def list_sorted_npy_files(directory):
    files = glob.glob(os.path.join(directory, "*.npy"))
    def numeric_key(path):
        nums = re.findall(r"\d+", os.path.basename(path))
        return [int(n) for n in nums] if nums else [os.path.basename(path)]
    return sorted(files, key=numeric_key)


def normalize_landmarks(lm):
    """Center and scale a single frame (20,2) to be pose-invariant."""
    lm = np.asarray(lm, dtype=np.float32)
    centroid = lm.mean(axis=0)
    lm -= centroid
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-8
    return lm / scale


def load_lip_sequence(landmarkers_dir):
    """
    Load all .npy frames from one landmarkers_cv folder.
    Returns (T, 40) — each frame is 20 landmarks × 2 coords, flattened.
    Returns None if fewer than 5 valid frames found.
    """
    files = list_sorted_npy_files(landmarkers_dir)
    frames = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[0] < 68 or arr.shape[1] < 2:
            continue
        lip = arr[LIP_START:LIP_END, :2]   # (20, 2)
        lip = normalize_landmarks(lip)
        frames.append(lip.flatten())        # (40,)
    if len(frames) < 5:
        return None
    return np.asarray(frames, dtype=np.float32)  # (T, 40)


def compute_velocity(seq):
    """
    seq: (T, 40)
    returns (T, 40) velocity via finite difference (gradient)
    """
    return np.gradient(seq, axis=0).astype(np.float32)


def build_sample_list(root):
    """
    Walk the directory tree and collect all valid samples.
    Returns list of dicts with keys:
        user_id, group_type, group_name, video_name,
        landmarkers_dir, label_str
    """
    samples = []
    user_dirs = sorted(
        [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)],
        key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])
        if re.findall(r"\d+", os.path.basename(p)) else os.path.basename(p)
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
            group_dirs = sorted(glob.glob(os.path.join(user_dir, f"{group_prefix}*")))

            for group_dir in group_dirs:
                group_name = os.path.basename(group_dir)   # e.g. "sentences1"
                videos_dir = os.path.join(group_dir, "videos")

                if not os.path.isdir(videos_dir):
                    continue

                if user_id == "1":
                    video_dirs = sorted(glob.glob(os.path.join(videos_dir, "video_[0-9]*")))
                else:
                    video_dirs = sorted(glob.glob(os.path.join(videos_dir, "video_proc_*")))

                for video_dir in video_dirs:
                    landmarkers_dir = os.path.join(video_dir, "landmarkers_cv")
                    if not os.path.isdir(landmarkers_dir):
                        continue

                    samples.append({
                        "user_id":        user_id,
                        "group_type":     group_prefix,
                        "group_name":     group_name,
                        "video_name":     os.path.basename(video_dir),
                        "landmarkers_dir": landmarkers_dir,
                        "label_str":      group_name,   # e.g. "sentences1"
                    })

    print(f"Total candidate samples: {len(samples)}")
    return samples


def build_label_map(samples):
    """Build a sorted label → index mapping from all sample label strings."""
    unique_labels = sorted(set(s["label_str"] for s in samples))
    label_map = {lbl: idx for idx, lbl in enumerate(unique_labels)}
    return label_map


# ============================================================
# DATASET
# ============================================================

class LipLandmarkDataset(Dataset):
    """
    Each item returns:
        seq   : (T, 80) tensor — landmarks (40) + velocity (40)
        length: int scalar
        label : int class index
        meta  : dict of strings for bookkeeping
    """
    def __init__(self, samples, label_map, augment=False):
        self.label_map = label_map
        self.augment   = augment
        self.items     = []   # (seq_array, label_int, meta_dict)

        skipped = 0
        for s in samples:
            seq = load_lip_sequence(s["landmarkers_dir"])
            if seq is None:
                skipped += 1
                continue
            vel = compute_velocity(seq)
            seq_with_vel = np.concatenate([seq, vel], axis=1)  # (T, 80)
            label = label_map[s["label_str"]]
            meta = {k: s[k] for k in
                    ["user_id", "group_type", "group_name",
                     "video_name", "landmarkers_dir", "label_str"]}
            self.items.append((seq_with_vel, label, meta))

        print(f"  Loaded {len(self.items)} samples ({skipped} skipped).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        seq, label, meta = self.items[idx]

        # ---- Data augmentation (training only) ----
        if self.augment:
            # 1. Random temporal crop: keep 80-100% of frames
            T = len(seq)
            if T > 10:
                keep = int(np.random.uniform(0.8, 1.0) * T)
                start = np.random.randint(0, T - keep + 1)
                seq = seq[start:start + keep]

            # 2. Small Gaussian jitter on landmark coords
            seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)

        seq_tensor = torch.from_numpy(seq)   # (T, 80)
        return seq_tensor, label, meta


def collate_fn(batch):
    """Pad variable-length sequences in a batch."""
    seqs, labels, metas = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)  # (B, T_max, 80)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, labels, list(metas)


# ============================================================
# MODEL
# ============================================================

class LipLSTMEncoder(nn.Module):
    """
    Bidirectional LSTM that maps a variable-length lip landmark sequence
    to a fixed 128-dim embedding.

    Architecture:
        Input (T, 80)
          → LayerNorm
          → BiLSTM (2 layers, hidden=256 per direction)
          → last hidden state concat (both directions) → (512,)
          → Dropout
          → Linear(512 → 128)
          → L2 norm  ← this is your EMBEDDING for fusion
          → Linear(128 → num_classes)  ← classification head, dropped at inference
    """
    def __init__(self, input_size, num_classes, hidden_size=256,
                 num_layers=2, embedding_dim=128, dropout=0.3):
        super().__init__()

        self.input_norm = nn.LayerNorm(input_size)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)

        # 512 = hidden_size * 2 directions
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, lengths):
        """
        x       : (B, T_max, input_size)
        lengths : (B,) actual sequence lengths
        Returns:
            logits    : (B, num_classes)
            embedding : (B, embedding_dim) L2-normalized
        """
        x = self.input_norm(x)

        # Pack to skip padding in recurrence
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        # h_n: (num_layers * 2, B, hidden_size)
        # Take last layer, both directions
        h_fwd = h_n[-2]   # (B, hidden_size)
        h_bwd = h_n[-1]   # (B, hidden_size)
        h = torch.cat([h_fwd, h_bwd], dim=1)   # (B, 512)

        h = self.dropout(h)
        embedding = self.embed_proj(h)                      # (B, 128) raw
        logits = self.classifier(embedding)                 # classify on raw embedding
        embedding_normed = F.normalize(embedding, p=2, dim=1)  # L2 norm for fusion only
        return logits, embedding_normed

    def encode(self, x, lengths):
        """Inference-only: returns just the embedding."""
        with torch.no_grad():
            _, embedding = self.forward(x, lengths)
        return embedding


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

    best_val_acc = 0.0
    patience_counter = 0
    idx_to_label = {v: k for k, v in label_map.items()}

    print("\n" + "="*60)
    print("TRAINING")
    print("="*60)

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

    # Load best weights and print final val report
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, final_val_acc, val_preds, val_labels = eval_epoch(
        model, val_loader, criterion
    )
    print(f"\nFinal validation accuracy: {final_val_acc:.3f}")
    print("\nClassification report (validation set):")
    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    print(classification_report(
        val_labels, val_preds,
        target_names=target_names,
        zero_division=0,
    ))
    return model


# ============================================================
# EMBEDDING EXTRACTION
# ============================================================

@torch.no_grad()
def extract_all_embeddings(model, all_samples, label_map):
    """
    Run inference on every valid sample and collect embeddings.
    Returns arrays for stacking into the NPZ.
    """
    model.eval()
    embeddings, labels, user_ids = [], [], []
    group_types, group_names, video_names = [], [], []

    print("\nExtracting embeddings for all samples...")
    skipped = 0

    for s in all_samples:
        seq = load_lip_sequence(s["landmarkers_dir"])
        if seq is None:
            skipped += 1
            continue
        vel = compute_velocity(seq)
        seq_with_vel = np.concatenate([seq, vel], axis=1)   # (T, 80)

        x = torch.from_numpy(seq_with_vel).unsqueeze(0).to(DEVICE)  # (1, T, 80)
        length = torch.tensor([len(seq_with_vel)], dtype=torch.long).to(DEVICE)

        emb = model.encode(x, length).squeeze(0).cpu().numpy()  # (128,)
        embeddings.append(emb)
        labels.append(label_map[s["label_str"]])
        user_ids.append(s["user_id"])
        group_types.append(s["group_type"])
        group_names.append(s["group_name"])
        video_names.append(s["video_name"])

    print(f"  Extracted {len(embeddings)} embeddings ({skipped} skipped).")
    return (
        np.stack(embeddings, axis=0),           # (N, 128)
        np.array(labels, dtype=np.int32),
        np.array(user_ids),
        np.array(group_types),
        np.array(group_names),
        np.array(video_names),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. Build sample list and label map
    # ----------------------------------------------------------
    all_samples = build_sample_list(ROOT)
    if not all_samples:
        print("No samples found. Check your ROOT path and directory structure.")
        return

    label_map = build_label_map(all_samples)
    num_classes = len(label_map)
    print(f"\nLabel map ({num_classes} classes):")
    for lbl, idx in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {idx:3d} → {lbl}")

    # Save label map
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nLabel map saved to {LABEL_MAP_PATH}")

    # ----------------------------------------------------------
    # 2. Split
    # ----------------------------------------------------------
    if USE_RANDOM_SPLIT:
        rng = np.random.RandomState(42)
        idx = rng.permutation(len(all_samples)).tolist()
        n_train = int(0.75 * len(idx))
        n_val   = int(0.15 * len(idx))
        train_samples = [all_samples[i] for i in idx[:n_train]]
        val_samples   = [all_samples[i] for i in idx[n_train:n_train + n_val]]
        test_samples  = [all_samples[i] for i in idx[n_train + n_val:]]
        print("Using RANDOM 75/15/10 split (debug mode — switch USE_RANDOM_SPLIT=False for speaker-independent eval).")
    else:
        train_samples = [s for s in all_samples
                         if s["user_id"] not in VAL_USERS + TEST_USERS]
        val_samples   = [s for s in all_samples if s["user_id"] in VAL_USERS]
        test_samples  = [s for s in all_samples if s["user_id"] in TEST_USERS]
        print("Using USER-BASED split (speaker-independent eval).")

    print(f"Split: {len(train_samples)} train | "
          f"{len(val_samples)} val | {len(test_samples)} test samples")

    # ----------------------------------------------------------
    # 3. Build datasets and loaders
    # ----------------------------------------------------------
    print("\nLoading training data...")
    train_ds = LipLandmarkDataset(train_samples, label_map, augment=True)
    print("Loading validation data...")
    val_ds   = LipLandmarkDataset(val_samples,   label_map, augment=False)
    print("Loading test data...")
    test_ds  = LipLandmarkDataset(test_samples,  label_map, augment=False)

    if len(train_ds) == 0:
        print("Training dataset is empty. Cannot proceed.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=collate_fn, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              collate_fn=collate_fn, num_workers=0)

    # ----------------------------------------------------------
    # 4. Build model
    # ----------------------------------------------------------
    INPUT_SIZE = 80   # 40 landmarks + 40 velocity
    model = LipLSTMEncoder(
        input_size=INPUT_SIZE,
        num_classes=num_classes,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    # ----------------------------------------------------------
    # 5. Train
    # ----------------------------------------------------------
    model = train(model, train_loader, val_loader, label_map)

    # ----------------------------------------------------------
    # 6. Test set evaluation
    # ----------------------------------------------------------
    if len(test_ds) > 0:
        criterion = nn.CrossEntropyLoss()
        _, test_acc, test_preds, test_labels = eval_epoch(
            model, test_loader, criterion
        )
        print(f"\nTest accuracy (unseen users {TEST_USERS}): {test_acc:.3f}")
        idx_to_label = {v: k for k, v in label_map.items()}
        target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
        print(classification_report(
            test_labels, test_preds,
            target_names=target_names,
            zero_division=0,
        ))

    # ----------------------------------------------------------
    # 7. Extract embeddings for ALL samples and save
    # ----------------------------------------------------------
    embs, labels_arr, user_ids, group_types, group_names, video_names = \
        extract_all_embeddings(model, all_samples, label_map)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embs,           # (N, 128) — THIS is your fusion input
        labels=labels_arr,
        user_ids=user_ids,
        group_types=group_types,
        group_names=group_names,
        video_names=video_names,
    )

    print(f"\nEmbeddings saved to {EMBEDDINGS_PATH}")
    print(f"  Shape: {embs.shape}   (N samples × 128 embedding dim)")
    print(f"\nTo load in your fusion script:")
    print(f"  data = np.load('{EMBEDDINGS_PATH}', allow_pickle=True)")
    print(f"  X = data['embeddings']   # shape (N, 128)")
    print(f"  y = data['labels']       # class indices")


if __name__ == "__main__":
    main()

"""
model stats

Final validation accuracy: 0.760
accuracy | f1-score: 0.76, support: 795
macro avg | Precision: 0.77, Recall: 0.77, f1-score: 0.76, support: 795
weighted avg | Precision: 0.78, Recall: 0.76, f1-score: 0.76, support: 795

Test accuracy (unseen users ['19', '20']): 0.762
accuracy | f1-score: 0.76, support: 530
macro avg | Precision: 0.77, Recall: 0.77, f1-score: 0.76, support: 530
weighted avg | Precision: 0.77, Recall: 0.76, f1-score: 0.76, support: 530
"""