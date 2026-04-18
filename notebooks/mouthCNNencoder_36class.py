"""
mouth_cnn_encoder.py
====================
CNN Encoder for Mouth Frame Images — Silent Speech Decoding
============================================================

Loads pre-extracted ResNet18 features from mouth_frame_embeddings.csv
(512-dim mean-pooled embeddings, one row per video) and trains a small
projection head to produce 128-dim embeddings matching the lip LSTM output.

Zero image loading — pure tensor operations only.

ARCHITECTURE
------------
  Input:  (N, 512) from CSV embed_0..embed_511
  Head:   Linear(512→256) + ReLU + LayerNorm + Dropout
          → Linear(256→128) + LayerNorm → L2-norm = embedding
  Train:  Linear(128→num_classes) classifier, dropped at inference

OUTPUTS
-------
- mouth_cnn_model.pt                 : best model weights
- mouth_frame_embeddings_trained.npz : embeddings for all samples
    embeddings  (N, 128)
    labels      (N,)   integer class indices
    users       (N,)   participant strings
    label_names (N,)   e.g. "word1", "word7"
- mouth_cnn_label_map.json
"""

import os
import json
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ============================================================
# CONFIG
# ============================================================

CSV_PATH        = "mouth_frame_embeddings.csv"
MODEL_PATH      = "mouth_cnn_model_36class.pt"
EMBEDDINGS_PATH = "mouth_frame_embeddings_trained_36class.npz"
LABEL_MAP_PATH  = "mouth_cnn_label_map_36class.json"

EMBEDDING_DIM  = 128
FRAME_FEAT_DIM = 256
DROPOUT        = 0.3

BATCH_SIZE = 64
LR         = 3e-4
EPOCHS     = 60
PATIENCE   = 20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ============================================================
# DATA
# ============================================================

def load_csv(path):
    print(f"Loading {path}...")
    df = pd.read_csv(path)

    embed_cols = [f"embed_{i}" for i in range(512)]
    X = df[embed_cols].values.astype(np.float32)        # (N, 512)

    labels_raw  = df["label_name"].values               # e.g. "sentences1"
    users       = df["participant"].astype(str).values
    video_names = df["video_name"].astype(str).values

    unique_labels = sorted(set(labels_raw))
    label_map     = {lbl: i for i, lbl in enumerate(unique_labels)}
    y             = np.array([label_map[l] for l in labels_raw], dtype=np.int32)

    print(f"  {len(df)} samples  |  {len(unique_labels)} classes  |  "
          f"feature dim: {X.shape[1]}")
    print(f"  Classes: {unique_labels}")
    return X, y, users, labels_raw, video_names, label_map


def make_loaders(X, y, seed=42):
    N   = len(X)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(N)

    n_train = int(0.80 * N)
    n_val   = int(0.10 * N)
    tr, va, te = idx[:n_train], idx[n_train:n_train+n_val], idx[n_train+n_val:]

    print(f"Split (seed={seed}): {len(tr)} train | {len(va)} val | {len(te)} test")

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)

    def loader(idxs, shuffle):
        ds = TensorDataset(Xt[idxs], yt[idxs])
        return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0)

    return loader(tr, True), loader(va, False), loader(te, False), tr, va, te


# ============================================================
# MODEL
# ============================================================

class ProjectionHead(nn.Module):
    """
    Linear(512→256→128) projection head trained on pre-extracted features.
    forward() → (logits, L2-normed 128-dim embedding)
    encode()  → 128-dim embedding only
    """
    def __init__(self, num_classes, in_dim=512,
                 feat_dim=256, embedding_dim=128, dropout=0.3):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(feat_dim),
            nn.Dropout(dropout),
            nn.Linear(feat_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x):
        emb    = self.proj(x)
        logits = self.classifier(emb)
        return logits, F.normalize(emb, p=2, dim=1)

    def encode(self, x):
        with torch.no_grad():
            _, emb = self.forward(x)
        return emb


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
        optimizer.step()
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        n          += len(labels)
    print(f"  Epoch {epoch:3d}/{total}  train  loss {total_loss/n:.4f}  acc {correct/n:.3f}",
          flush=True)
    return total_loss / n, correct / n


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_labels = [], []
    for x, labels in loader:
        x, labels = x.to(DEVICE), labels.to(DEVICE)
        logits, _ = model(x)
        loss = criterion(logits, labels)
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
    X, y, users, label_names, video_names, label_map = load_csv(CSV_PATH)
    num_classes = len(label_map)

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    train_loader, val_loader, test_loader, tr_idx, va_idx, te_idx = \
        make_loaders(X, y)

    model = ProjectionHead(
        num_classes=num_classes,
        embedding_dim=EMBEDDING_DIM,
        feat_dim=FRAME_FEAT_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable:,}\n")

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

    results = {
        "num_classes": num_classes,
        "n_samples": len(X),
        "split": {"train": int(0.80 * len(X)), "val": int(0.10 * len(X))},
        "val_accuracy": round(val_acc, 4),
        "test_accuracy": round(test_acc, 4),
        "val_report": val_report,
        "test_report": test_report,
    }
    results_path = MODEL_PATH.replace(".pt", "_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Extract embeddings for all samples
    model.eval()
    Xt = torch.tensor(X, dtype=torch.float32).to(DEVICE)
    all_embs = []
    with torch.no_grad():
        for start in range(0, len(Xt), BATCH_SIZE):
            chunk = Xt[start:start + BATCH_SIZE]
            all_embs.append(model.encode(chunk).cpu().numpy())
    all_embs = np.concatenate(all_embs, axis=0)   # (N, 128)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=all_embs,
        labels=y,
        users=users,
        label_names=label_names,
        video_names=video_names,
    )

    print(f"\nEmbeddings saved to {EMBEDDINGS_PATH}  shape: {all_embs.shape}")
    print(f"\nTo load:")
    print(f"  data = np.load('{EMBEDDINGS_PATH}', allow_pickle=True)")
    print(f"  X    = data['embeddings']   # (N, 128)")
    print(f"  y    = data['labels']")


if __name__ == "__main__":
    main()
