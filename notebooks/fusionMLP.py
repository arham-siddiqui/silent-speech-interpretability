"""
fusionMLP.py
============
Multimodal attention-weighted fusion for silent speech decoding.

WHAT THIS FILE DOES
-------------------
1. Loads 128-dim embeddings from each modality NPZ
2. Aligns them by (user_id, group_name) via mean-pooling over repetitions
   — each (speaker, utterance) pair becomes one fused sample
3. Trains an attention-weighted fusion model:
     K modalities × 128-dim → per-sample attention (K,) → attended (128,) → MLP → class
4. Saves the fusion model, fused embeddings, and an attention heatmap showing
   which modality the model relied on per utterance type

WHY ATTENTION OVER CONCATENATION
---------------------------------
Simple concat → MLP (K×128 → classes) forces the MLP to figure out weighting
implicitly. Attention makes it explicit and interpretable:
  - You can read off "for vowels, the model trusted lip 40% / laser 35% / radar 25%"
  - That's a publishable result — which sensors are informative for which speech acts
  - Also enables modality dropout training: randomly zero one modality per sample
    so the model learns to operate with incomplete sensor sets

ALIGNMENT NOTE
--------------
Each modality records N repetitions per (user, utterance). The repetitions
are not time-locked across modalities, so we CANNOT align by index.
Instead, we mean-pool all repetitions within (user_id, group_name) → one
representative embedding per pair. This gives ~553 aligned samples (the
intersection across modalities — radar is missing user 11).

TO ADD A 5TH MODALITY
----------------------
Append to NPZ_FILES dict below. Nothing else changes.

OUTPUTS
-------
- fusion_model.pt               : trained model weights
- fusion_embeddings.npz         : attended 128-dim representations for all aligned samples
- fusion_label_map.json         : class index → label string
- attention_weights_per_class.png : heatmap of mean attention weight per (class, modality)
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.metrics import classification_report

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib not found — attention plot will be skipped")

# ============================================================
# CONFIG
# ============================================================

NPZ_FILES = {
    "lip":   "lip_embeddings.npz",
    "laser": "laser_embeddings.npz",
    "radar": "radar_embeddings.npz",
    # "mmwave": "mmwave_embeddings.npz",   # ← uncomment when available
    # "modality5": "modality5_embeddings.npz",
}

# Speaker-independent split (same convention as per-modality scripts)
VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]

# Model
EMBEDDING_DIM = 128
HIDDEN_SIZES  = [256, 128]  # MLP layers after attention pooling
DROPOUT       = 0.3

# Training
BATCH_SIZE           = 32
LR                   = 3e-4
EPOCHS               = 150
PATIENCE             = 25
MODALITY_DROP_PROB   = 0.20  # prob of zeroing one random modality per training sample
EMBED_NOISE_STD      = 0.01  # small Gaussian noise added to embeddings during training

# Output paths
MODEL_PATH      = "fusion_model.pt"
EMBEDDINGS_PATH = "fusion_embeddings.npz"
LABEL_MAP_PATH  = "fusion_label_map.json"
ATTN_PLOT_PATH  = "attention_weights_per_class.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# DATA LOADING AND ALIGNMENT
# ============================================================

def load_and_align(npz_files: dict):
    """
    Load each modality NPZ, mean-pool over repetitions within (user_id, group_name),
    then return only samples present in ALL modalities.

    Returns:
        X      : np.ndarray (N, K, 128) — N aligned samples, K modalities
        y      : np.ndarray (N,)        — integer class labels
        users  : np.ndarray (N,)        — user_id strings
        groups : np.ndarray (N,)        — group_name strings (e.g. "sentences1")
        label_map : dict[str → int]
        modality_names : list[str]
    """
    modality_names = list(npz_files.keys())
    pooled = {}  # modality_name → dict[(user_id, group_name) → mean embedding]

    for name, path in npz_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ not found: {path}")
        d = np.load(path, allow_pickle=True)

        # Group embeddings by (user_id, group_name)
        groups_raw = defaultdict(list)
        user_ids   = d["user_ids"].astype(str)
        grp_names  = d["group_names"].astype(str)
        embs       = d["embeddings"]  # (N, 128)

        for i in range(len(embs)):
            key = (user_ids[i], grp_names[i])
            groups_raw[key].append(embs[i])

        # Mean-pool repetitions → one vector per (user, utterance)
        pooled[name] = {k: np.mean(v, axis=0) for k, v in groups_raw.items()}
        print(f"  {name:12s}: {len(pooled[name])} unique (user, group) pairs "
              f"loaded from {os.path.basename(path)}")

    # Intersection of keys across all modalities
    all_keys = None
    for p in pooled.values():
        s = set(p.keys())
        all_keys = s if all_keys is None else all_keys & s
    all_keys = sorted(all_keys)  # deterministic order

    N = len(all_keys)
    K = len(modality_names)
    print(f"\n  Intersection: {N} aligned (user, group) pairs "
          f"across {K} modalities")

    # Stack into (N, K, 128)
    X = np.zeros((N, K, 128), dtype=np.float32)
    for j, name in enumerate(modality_names):
        for i, key in enumerate(all_keys):
            X[i, j] = pooled[name][key]

    # Build a fresh unified label map from group names
    unique_groups = sorted(set(g for _, g in all_keys))
    label_map = {g: i for i, g in enumerate(unique_groups)}

    y      = np.array([label_map[g] for _, g in all_keys], dtype=np.int32)
    users  = np.array([u for u, _ in all_keys])
    groups = np.array([g for _, g in all_keys])

    return X, y, users, groups, label_map, modality_names


# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):
    """
    X:     (N, K, 128) — one row per modality per sample
    y:     (N,)        — integer class labels
    users: (N,)        — user_id strings (for reference / split checking)
    """
    def __init__(self, X, y, users, augment=False):
        self.X      = torch.from_numpy(X).float()     # (N, K, 128)
        self.y      = torch.from_numpy(y).long()      # (N,)
        self.users  = users
        self.augment = augment
        self.K = X.shape[1]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].clone()     # (K, 128)
        label = self.y[idx]

        if self.augment:
            # Modality dropout: zero out one random modality
            if np.random.rand() < MODALITY_DROP_PROB:
                k = np.random.randint(self.K)
                x[k] = 0.0

            # Small embedding noise (prevents over-reliance on exact embedding values)
            x = x + torch.randn_like(x) * EMBED_NOISE_STD

        return x, label


# ============================================================
# MODEL
# ============================================================

class AttentionFusion(nn.Module):
    """
    Per-sample attention over K modality embeddings, then MLP classifier.

    Architecture:
        Input: (B, K, 128)
        ↓ per-modality LayerNorm
        ↓ attention scorer: Linear(128→64)→Tanh→Linear(64→1) per modality
        ↓ softmax over K → weights (B, K) summing to 1
        ↓ weighted sum → attended (B, 128)
        ↓ MLP: Linear→LayerNorm→GELU→Dropout × n_layers
        ↓ Linear → (B, num_classes)

    The attention weights are returned so you can log which modality
    was trusted most per sample or per utterance type.

    Notes:
    - LayerNorm per modality handles the fact that embeddings from different
      encoders may have different scales (especially if some were trained with
      random split vs user split)
    - The attention scorer takes each 128-dim embedding as input, so weights
      adapt per sample rather than being fixed global weights
    - Modality dropout during training forces the model to not collapse onto
      one modality, making it robust to missing sensors at inference time
    """

    def __init__(self, num_modalities: int, embedding_dim: int = 128,
                 num_classes: int = 30, hidden_sizes=(256, 128), dropout: float = 0.3):
        super().__init__()
        self.num_modalities = num_modalities
        self.embedding_dim  = embedding_dim

        # Independent LayerNorm per modality (handles scale differences between encoders)
        self.modality_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_modalities)
        ])

        # Attention scorer: maps each 128-dim embedding to a scalar score
        # Shared weights across modalities (they all live in the same embedding space)
        self.attn_scorer = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1, bias=False),
        )

        # Classifier MLP operating on the attended 128-dim representation
        layers = []
        in_dim = embedding_dim
        for h in hidden_sizes:
            layers += [
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):
        """
        x: (B, K, embedding_dim)

        Returns:
            logits  : (B, num_classes)
            weights : (B, K)  attention weights, sum to 1 per sample
        """
        B, K, D = x.shape

        # Per-modality layer norm
        x = torch.stack(
            [self.modality_norms[i](x[:, i, :]) for i in range(K)], dim=1
        )  # (B, K, D)

        # Attention scores and weights
        scores  = self.attn_scorer(x).squeeze(-1)   # (B, K)
        weights = F.softmax(scores, dim=1)           # (B, K)  sums to 1

        # Weighted sum over modalities → (B, D)
        attended = (weights.unsqueeze(-1) * x).sum(dim=1)

        logits = self.classifier(attended)
        return logits, weights

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """Returns the attended 128-dim embedding (before classifier)."""
        B, K, D = x.shape
        x = torch.stack(
            [self.modality_norms[i](x[:, i, :]) for i in range(K)], dim=1
        )
        scores  = self.attn_scorer(x).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        attended = (weights.unsqueeze(-1) * x).sum(dim=1)
        return attended, weights


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits, _ = model(x)
        loss = criterion(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item() * len(y)
        correct    += (logits.argmax(1) == y).sum().item()
        total      += len(y)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_weights = [], [], []
    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits, weights = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)
        preds = logits.argmax(1)
        correct    += (preds == y).sum().item()
        total      += len(y)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_weights.append(weights.cpu().numpy())
    all_weights = np.concatenate(all_weights, axis=0) if all_weights else np.array([])
    return total_loss / total, correct / total, all_preds, all_labels, all_weights


def run_training(model, train_loader, val_loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    patience_ctr = 0

    print("\n" + "=" * 60)
    print("TRAINING FUSION MODEL")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion)
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
    return model


# ============================================================
# ATTENTION VISUALIZATION
# ============================================================

def plot_attention_by_class(weights, labels, label_map, modality_names, path):
    """
    Heatmap: rows = utterance classes, cols = modalities.
    Cell value = mean attention weight the model assigned to that modality
    when classifying that utterance type.

    High value → the model relied heavily on this modality for this class.
    """
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping attention plot (matplotlib not available)")
        return

    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes = len(label_map)
    K = len(modality_names)

    mean_w = np.zeros((num_classes, K))
    for cls in range(num_classes):
        mask = np.array(labels) == cls
        if mask.sum() > 0:
            mean_w[cls] = weights[mask].mean(axis=0)

    fig_h = max(8, num_classes * 0.35)
    fig_w = max(5, K * 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mean_w, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(K))
    ax.set_xticklabels(modality_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([idx_to_label[i] for i in range(num_classes)], fontsize=8)
    ax.set_xlabel("Modality", fontsize=12)
    ax.set_ylabel("Utterance class", fontsize=12)
    ax.set_title("Mean attention weight per utterance class × modality\n"
                 "(darker = model relied on this modality more)", fontsize=11)

    # Annotate cells with values
    for r in range(num_classes):
        for c in range(K):
            ax.text(c, r, f"{mean_w[r, c]:.2f}",
                    ha="center", va="center", fontsize=6,
                    color="white" if mean_w[r, c] > 0.6 else "black")

    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention heatmap saved to: {path}")


def print_attention_summary(weights, labels, label_map, modality_names):
    """Print mean attention per modality, broken down by utterance group type."""
    idx_to_label = {v: k for k, v in label_map.items()}
    label_strings = np.array([idx_to_label[l] for l in labels])

    print("\nMean attention weights per modality:")
    print(f"  {'Modality':<12} {'Overall':>8}", end="")

    group_types = ["sentences", "vowel", "word"]
    for gt in group_types:
        print(f"  {gt:>10}", end="")
    print()

    overall_weights = weights.mean(axis=0)
    for j, name in enumerate(modality_names):
        print(f"  {name:<12} {overall_weights[j]:>8.3f}", end="")
        for gt in group_types:
            mask = np.array([gt in s for s in label_strings])
            w = weights[mask, j].mean() if mask.sum() > 0 else float("nan")
            print(f"  {w:>10.3f}", end="")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    # ----------------------------------------------------------
    # 1. Load and align modalities
    # ----------------------------------------------------------
    print("Loading and aligning modalities...")
    X, y, users, groups, label_map, modality_names = load_and_align(NPZ_FILES)

    num_classes    = len(label_map)
    num_modalities = len(modality_names)
    N              = len(X)

    print(f"\n{num_classes} classes | {num_modalities} modalities: {modality_names}")
    print(f"Total aligned samples: {N}")
    print(f"Chance accuracy: 1/{num_classes} = {1/num_classes:.3f}")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to {LABEL_MAP_PATH}")

    # ----------------------------------------------------------
    # 2. Speaker-independent split
    # ----------------------------------------------------------
    train_mask = ~np.isin(users, VAL_USERS + TEST_USERS)
    val_mask   = np.isin(users, VAL_USERS)
    test_mask  = np.isin(users, TEST_USERS)

    print(f"\nSplit (user-based, speaker-independent):")
    print(f"  Train: {train_mask.sum()} samples  (users excluding {VAL_USERS + TEST_USERS})")
    print(f"  Val:   {val_mask.sum()} samples  (users {VAL_USERS})")
    print(f"  Test:  {test_mask.sum()} samples  (users {TEST_USERS})")

    # NOTE: lip_embeddings.npz was trained with random split (speaker leakage).
    # Val/test accuracy will be inflated until lip is retrained with USE_RANDOM_SPLIT=False
    # and lip_embeddings.npz is regenerated. The fusion split itself is correct here.

    train_ds = FusionDataset(X[train_mask], y[train_mask], users[train_mask], augment=True)
    val_ds   = FusionDataset(X[val_mask],   y[val_mask],   users[val_mask],   augment=False)
    test_ds  = FusionDataset(X[test_mask],  y[test_mask],  users[test_mask],  augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------
    # 3. Build model
    # ----------------------------------------------------------
    model = AttentionFusion(
        num_modalities=num_modalities,
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        hidden_sizes=HIDDEN_SIZES,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nFusion model parameters: {total_params:,}")
    print(f"Input: ({num_modalities}, {EMBEDDING_DIM}) → attention → ({EMBEDDING_DIM},) → {num_classes} classes")

    # ----------------------------------------------------------
    # 4. Train
    # ----------------------------------------------------------
    model = run_training(model, train_loader, val_loader)

    # ----------------------------------------------------------
    # 5. Evaluate on test set
    # ----------------------------------------------------------
    criterion = nn.CrossEntropyLoss()
    _, test_acc, test_preds, test_labels, test_weights = eval_epoch(
        model, test_loader, criterion
    )
    print(f"\nTest accuracy (users {TEST_USERS}): {test_acc:.3f}")
    idx_to_label = {v: k for k, v in label_map.items()}
    print(classification_report(
        test_labels, test_preds,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        zero_division=0,
    ))

    # ----------------------------------------------------------
    # 6. Attention analysis (val + test, more data for stable means)
    # ----------------------------------------------------------
    _, val_acc, _, val_labels, val_weights = eval_epoch(model, val_loader, criterion)
    print(f"Val accuracy: {val_acc:.3f}")

    eval_weights = np.concatenate([val_weights, test_weights], axis=0)
    eval_labels  = val_labels + test_labels

    print_attention_summary(eval_weights, eval_labels, label_map, modality_names)

    plot_attention_by_class(
        eval_weights, eval_labels, label_map, modality_names, ATTN_PLOT_PATH
    )

    # ----------------------------------------------------------
    # 7. Save fused embeddings for all aligned samples
    # ----------------------------------------------------------
    all_ds     = FusionDataset(X, y, users, augment=False)
    all_loader = DataLoader(all_ds, batch_size=64, shuffle=False)

    model.eval()
    fused_embs, all_weights_full, all_preds_full = [], [], []
    with torch.no_grad():
        for x_batch, _ in all_loader:
            x_batch = x_batch.to(DEVICE)
            attended, w = model.encode(x_batch)
            fused_embs.append(attended.cpu().numpy())
            all_weights_full.append(w.cpu().numpy())
            logits, _ = model(x_batch)
            all_preds_full.extend(logits.argmax(1).cpu().numpy())

    fused_embs       = np.concatenate(fused_embs, axis=0)        # (N, 128)
    all_weights_full = np.concatenate(all_weights_full, axis=0)  # (N, K)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=fused_embs,               # (N, 128) — attended fusion representation
        labels=y,                             # (N,)
        users=users,                          # (N,)
        groups=groups,                        # (N,)
        attention_weights=all_weights_full,   # (N, K) — per-sample modality weights
        modality_names=np.array(modality_names),
    )
    print(f"\nFused embeddings saved to {EMBEDDINGS_PATH}")
    print(f"  embeddings shape:        {fused_embs.shape}")
    print(f"  attention_weights shape: {all_weights_full.shape}")
    print(f"\nTo load:")
    print(f"  d = np.load('{EMBEDDINGS_PATH}', allow_pickle=True)")
    print(f"  X = d['embeddings']           # (N, 128) fused representation")
    print(f"  W = d['attention_weights']    # (N, K) — which modality was trusted")


if __name__ == "__main__":
    main()
