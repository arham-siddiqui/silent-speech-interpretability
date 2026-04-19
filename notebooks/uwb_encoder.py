"""
uwb_encoder.py
==============
CNN Encoder for UWB Radar Data — Silent Speech Decoding
========================================================

Parallel to mouthCNNencoder.py but reads raw .npy files instead of a CSV.

ARCHITECTURE
------------
  Input:  (1, 205, 256) — per-sample normalised, time-padded UWB range-time map
  CNN:    Conv(1→32) + BN + ReLU + MaxPool
          Conv(32→64) + BN + ReLU + MaxPool
          Conv(64→128) + BN + ReLU + AdaptiveAvgPool(4,4)
          Flatten → (2048,)
          Linear(2048→256) + ReLU + LN + Dropout(0.3)
          Linear(256→128) + LN → L2-norm = 128-dim embedding
  Head:   Linear(128→num_classes)  [dropped at inference]

OUTPUTS
-------
- uwb_cnn_model.pt
- uwb_embeddings.npz
    embeddings   (N, 128)
    labels       (N,)
    user_ids     (N,)
    group_names  (N,)
    sample_names (N,)
- uwb_label_map.json
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
from sklearn.metrics import classification_report

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# CONFIG
# ============================================================

ROOT = "src/data/RVTALL/Processed_cut_data/uwb_processed/"

MODEL_PATH      = "uwb_cnn_model.pt"
EMBEDDINGS_PATH = "uwb_embeddings.npz"
LABEL_MAP_PATH  = "uwb_label_map.json"

TARGET_H = 205
TARGET_W = 256
EMBEDDING_DIM = 128
DROPOUT       = 0.3

BATCH_SIZE = 32
LR         = 3e-4
EPOCHS     = 60
PATIENCE   = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]


# ============================================================
# DATA LOADING
# ============================================================

def numeric_key(name):
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 0


def build_sample_list(root):
    samples = []
    skipped = 0

    user_dirs = sorted(
        [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)],
        key=lambda p: numeric_key(os.path.basename(p))
    )
    if not user_dirs:
        raise RuntimeError(f"No user dirs found under: {os.path.abspath(root)}")

    print(f"Found {len(user_dirs)} user directories.")

    for user_dir in user_dirs:
        user_id = os.path.basename(user_dir)

        cat_dirs = []
        for prefix in ["sentences_*", "vowel_*", "word_*"]:
            cat_dirs += glob.glob(os.path.join(user_dir, prefix))
        for cat_dir in sorted(cat_dirs, key=lambda p: os.path.basename(p)):
            group_name = os.path.basename(cat_dir)

            for fpath in sorted(glob.glob(os.path.join(cat_dir, "*.npy"))):
                try:
                    arr = np.load(fpath)
                    if arr.ndim != 2:
                        skipped += 1
                        continue
                except Exception:
                    skipped += 1
                    continue

                samples.append({
                    "user_id":     user_id,
                    "group_name":  group_name,
                    "sample_name": os.path.splitext(os.path.basename(fpath))[0],
                    "file_path":   fpath,
                    "label_str":   group_name,
                })

    print(f"Loaded {len(samples)} samples ({skipped} skipped).")
    return samples


def build_label_map(samples):
    unique = sorted(set(s["label_str"] for s in samples),
                    key=lambda x: numeric_key(x))
    return {lbl: i for i, lbl in enumerate(unique)}


def load_tensor(file_path):
    """Load one .npy file, normalise, resize to (1, 205, 256)."""
    arr = np.load(file_path).astype(np.float32)   # (205, T)
    mean, std = arr.mean(), arr.std()
    arr = (arr - mean) / (std + 1e-8)

    t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1, 1, 205, T)
    t = F.interpolate(t, size=(TARGET_H, TARGET_W),
                      mode="bilinear", align_corners=False)  # (1, 1, 205, 256)
    return t.squeeze(0)   # (1, 205, 256)


# ============================================================
# DATASET
# ============================================================

class UWBDataset(Dataset):
    def __init__(self, samples, label_map, augment=False):
        self.augment = augment
        self.items   = []
        skipped = 0

        for s in samples:
            try:
                tensor = load_tensor(s["file_path"])   # (1, 205, 256)
            except Exception:
                skipped += 1
                continue
            label = label_map[s["label_str"]]
            self.items.append((tensor, label, s))

        print(f"  Dataset: {len(self.items)} samples ({skipped} skipped).")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tensor, label, meta = self.items[idx]
        if self.augment:
            # Small additive noise
            tensor = tensor + torch.randn_like(tensor) * 0.02
        return tensor, label


# ============================================================
# MODEL
# ============================================================

class UWBCNNEncoder(nn.Module):
    """
    Small 2D CNN trained from scratch on (1, 205, 256) UWB range-time maps.
    encode() → 128-dim L2-normed embedding.
    """
    def __init__(self, num_classes, embedding_dim=128, dropout=0.3):
        super().__init__()

        self.cnn = nn.Sequential(
            # Block 1: (1, 205, 256) → (32, 102, 128)
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 2: (32, 102, 128) → (64, 51, 64)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Block 3: (64, 51, 64) → (128, 4, 4)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.head = nn.Sequential(
            nn.Flatten(),                          # (2048,)
            nn.Linear(2048, 256),
            nn.ReLU(inplace=True),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        feat   = self.cnn(x)
        emb    = self.head(feat)
        logits = self.classifier(emb)
        return logits, F.normalize(emb, p=2, dim=1)

    def encode(self, x):
        with torch.no_grad():
            _, emb = self.forward(x)
        return emb


# ============================================================
# SPLIT
# ============================================================

def make_split(dataset):
    tr_items, va_items, te_items = [], [], []
    for item in dataset.items:
        uid = item[2]["user_id"]
        if uid in VAL_USERS:
            va_items.append(item)
        elif uid in TEST_USERS:
            te_items.append(item)
        else:
            tr_items.append(item)
    print(f"Split (user-based): {len(tr_items)} train | "
          f"{len(va_items)} val | {len(te_items)} test")
    print(f"  Val users: {VAL_USERS}  Test users: {TEST_USERS}")
    return tr_items, va_items, te_items


class SubsetFromItems(Dataset):
    def __init__(self, items, augment=False):
        self.items   = items
        self.augment = augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        tensor, label, _ = self.items[idx]
        if self.augment:
            tensor = tensor + torch.randn_like(tensor) * 0.02
        return tensor, label


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion, epoch, total):
    model.train()
    total_loss, correct, n = 0.0, 0, 0
    for x, labels in loader:
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    print(f"  Epoch {epoch:3d}/{total}  train  loss {total_loss/n:.4f}  "
          f"acc {correct/n:.3f}", flush=True)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, labels in loader:
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(x)
        loss       = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        preds       = logits.argmax(1)
        correct    += (preds == labels).sum().item()
        n          += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    return total_loss / n, correct / n, all_preds, all_labels


def train(model, train_loader, val_loader, label_map):
    criterion    = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer    = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler    = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    idx_to_label = {v: k for k, v in label_map.items()}

    best_val_acc = 0.0
    patience_ctr = 0

    print("\n" + "=" * 55)
    print("TRAINING")
    print("=" * 55)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(
            model, train_loader, optimizer, criterion, epoch, EPOCHS
        )
        val_loss, val_acc, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step()

        print(f"           val   loss {val_loss:.4f}  acc {val_acc:.3f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Best val acc: {best_val_acc:.3f} — saved.")
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
    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    val_report = classification_report(val_labels, val_preds,
                                       target_names=target_names,
                                       zero_division=0, output_dict=True)
    print(classification_report(val_labels, val_preds,
                                target_names=target_names, zero_division=0))
    return model, final_val_acc, val_report


# ============================================================
# MAIN
# ============================================================

def main():
    all_samples = build_sample_list(ROOT)
    if not all_samples:
        print("No samples found. Check ROOT.")
        return

    label_map   = build_label_map(all_samples)
    num_classes = len(label_map)
    print(f"\nLabel map ({num_classes} classes):")
    for lbl, idx in sorted(label_map.items(), key=lambda x: x[1]):
        print(f"  {idx:3d} → {lbl}")
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    # Load all into memory
    print("\nLoading all samples into memory...")
    full_ds = UWBDataset(all_samples, label_map, augment=False)
    if len(full_ds) == 0:
        print("Dataset is empty.")
        return

    tr_items, va_items, te_items = make_split(full_ds)

    train_loader = DataLoader(SubsetFromItems(tr_items, augment=True),
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader   = DataLoader(SubsetFromItems(va_items),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader  = DataLoader(SubsetFromItems(te_items),
                              batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = UWBCNNEncoder(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters: {total_params:,}\n")

    model, val_acc, val_report = train(model, train_loader, val_loader, label_map)

    # Test evaluation
    criterion = nn.CrossEntropyLoss()
    _, test_acc, test_preds, test_labels_arr = eval_epoch(
        model, test_loader, criterion
    )
    print(f"\nTest accuracy: {test_acc:.3f}")
    idx_to_label = {v: k for k, v in label_map.items()}
    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    test_report = classification_report(test_labels_arr, test_preds,
                                        target_names=target_names,
                                        zero_division=0, output_dict=True)
    print(classification_report(test_labels_arr, test_preds,
                                target_names=target_names, zero_division=0))

    # Save results JSON
    results = {
        "num_classes": num_classes,
        "n_samples": len(full_ds),
        "split": {"train_users": "1-16", "val_users": "17-18",
                  "test_users": "19-20",
                  "train_n": len(tr_items), "val_n": len(va_items),
                  "test_n": len(te_items)},
        "val_accuracy":  round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "val_report":  val_report,
        "test_report": test_report,
    }
    results_path = MODEL_PATH.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Extract embeddings for all samples
    print("\nExtracting embeddings for all samples...")
    all_loader = DataLoader(full_ds, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=0)
    model.eval()
    all_embs = []
    with torch.no_grad():
        for x, _ in all_loader:
            all_embs.append(model.encode(x.to(DEVICE)).cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)   # (N, 128)

    # Metadata aligned with full_ds.items order
    labels_arr      = np.array([item[1] for item in full_ds.items], dtype=np.int32)
    user_ids_arr    = np.array([item[2]["user_id"]     for item in full_ds.items])
    group_names_arr = np.array([item[2]["group_name"]  for item in full_ds.items])
    sample_names_arr= np.array([item[2]["sample_name"] for item in full_ds.items])

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=all_embs,
        labels=labels_arr,
        user_ids=user_ids_arr,
        group_names=group_names_arr,
        sample_names=sample_names_arr,
    )

    print(f"Embeddings saved to {EMBEDDINGS_PATH}  shape: {all_embs.shape}")
    print(f"\nTo load:")
    print(f"  data = np.load('{EMBEDDINGS_PATH}', allow_pickle=True)")
    print(f"  X    = data['embeddings']   # (N, 128)")
    print(f"  y    = data['labels']")


if __name__ == "__main__":
    main()
