"""
fusionMLP.py
============
Multimodal Transformer fusion for silent speech decoding.

WHAT THIS FILE DOES
-------------------
1. Loads 128-dim embeddings from each modality NPZ
2. Aligns by (user_id, group_name) and expands training data by pairing
   individual repetitions by position (not mean-pooling for train).
   Val/test use mean-pooled embeddings for stable evaluation.
3. Trains a Transformer encoder over K modality tokens:
     K × 128-dim tokens → self-attention (cross-modal) → mean-pool → MLP
4. Saves the fusion model, fused embeddings, and an attention heatmap.

WHY TRANSFORMER OVER SIMPLE ATTENTION
--------------------------------------
Simple weighted sum (previous approach) computes each modality's weight
independently — it cannot learn "if lip says X, trust radar more." The
Transformer self-attention lets every modality token attend to every other,
learning conditional cross-modal interactions. It also produces interpretable
per-head attention patterns that show modality relationships.

WHY INDIVIDUAL REPETITIONS FOR TRAINING
-----------------------------------------
Mean-pooling gives only ~420 training samples (1 per speaker × utterance).
With 30 classes, that's ~14 samples per class — not enough for any model.
By pairing the k-th repetition of each modality (sorted by index within
each group), we get ~8x more training data (~3600+ samples). Val/test still
use mean-pooled embeddings for deterministic evaluation.

TO ADD A MODALITY
-----------------
1. Add to NPZ_FILES dict
2. Add to NPZ_KEY_MAP if its field names differ from "user_ids"/"group_names"

OUTPUTS
-------
- fusion_model.pt
- fusion_embeddings.npz  (mean-pooled aligned embeddings with transformer output)
- fusion_label_map.json
- attention_weights_per_class.png
"""

import os
import re
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
    "uwb":   "uwb_embeddings.npz",
    "mouth": "mouth_frame_embeddings_trained_36class.npz",
}

# Map modality → (user_id_key, group_name_key) for NPZs with non-standard fields
NPZ_KEY_MAP = {
    "mouth": ("users", "label_names"),
}

VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]

# Model
EMBEDDING_DIM = 128
N_HEADS       = 4     # attention heads (128 / 4 = 32 per head)
N_LAYERS      = 2     # transformer encoder layers
DROPOUT       = 0.3

# Training
BATCH_SIZE         = 64
LR                 = 3e-4
EPOCHS             = 200
PATIENCE           = 30
WEIGHT_DECAY       = 1e-2
MODALITY_DROP_PROB = 0.20   # prob of zeroing one random modality per train sample
EMBED_NOISE_STD    = 0.01

# Output paths
MODEL_PATH      = "fusion_model.pt"
EMBEDDINGS_PATH = "fusion_embeddings.npz"
LABEL_MAP_PATH  = "fusion_label_map.json"
ATTN_PLOT_PATH  = "attention_weights_per_class.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# DATA LOADING
# ============================================================

def _natural_sort_key(s: str):
    """Sort strings containing numbers in natural order ('sample9' < 'sample10')."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_all_repetitions(npz_files: dict) -> dict:
    """
    Load every NPZ and group embeddings by (user_id, group_name),
    sorting repetitions within each group by their sample/video name.

    Returns:
        reps[modality_name][(user_id, group_name)] = [emb_0, emb_1, ..., emb_N]
        (each emb_i is a (128,) array, sorted by sample index)
    """
    reps = {}
    for name, path in npz_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ not found: {path}")
        d = np.load(path, allow_pickle=True)

        user_key, group_key = NPZ_KEY_MAP.get(name, ("user_ids", "group_names"))
        user_ids  = d[user_key].astype(str)
        grp_names = d[group_key].astype(str)
        embs      = d["embeddings"]  # (N, 128)

        # Find a field to sort by within each group
        sort_vals = None
        for k in ["sample_names", "video_names"]:
            if k in d.files:
                sort_vals = d[k].astype(str)
                break

        groups = defaultdict(list)
        for i in range(len(embs)):
            key = (user_ids[i], grp_names[i])
            sv  = sort_vals[i] if sort_vals is not None else str(i)
            groups[key].append((sv, embs[i]))

        reps[name] = {
            key: [e for _, e in sorted(items, key=lambda x: _natural_sort_key(x[0]))]
            for key, items in groups.items()
        }

        n_pairs = len(reps[name])
        n_total = sum(len(v) for v in reps[name].values())
        print(f"  {name:12s}: {n_pairs} (user, group) pairs, "
              f"{n_total} total reps — {os.path.basename(path)}")

    return reps


def build_aligned_splits(reps: dict, modality_names: list):
    """
    Find common (user, group) keys across all modalities.
    Training split: expand each pair into min-count aligned samples by position.
    Val/test split: mean-pool repetitions → one stable sample per pair.

    Returns:
        train_X : (N_train, K, 128)
        val_X   : (N_val,   K, 128)  — mean-pooled
        test_X  : (N_test,  K, 128)  — mean-pooled
        ... and corresponding y, users arrays
        label_map : dict[str → int]
    """
    # Intersection of keys across all modalities
    common_keys = None
    for name in modality_names:
        s = set(reps[name].keys())
        common_keys = s if common_keys is None else common_keys & s
    common_keys = sorted(common_keys)

    print(f"\n  Intersection: {len(common_keys)} (user, group) pairs across {len(modality_names)} modalities")

    unique_groups = sorted(set(g for _, g in common_keys))
    label_map     = {g: i for i, g in enumerate(unique_groups)}

    train_X, train_y, train_u = [], [], []
    val_X,   val_y,   val_u   = [], [], []
    test_X,  test_y,  test_u  = [], [], []

    for user, group in common_keys:
        label = label_map[group]

        if user in VAL_USERS or user in TEST_USERS:
            # Mean-pool for stable evaluation
            mean_stack = np.stack([
                np.mean(reps[m][(user, group)], axis=0) for m in modality_names
            ], axis=0)  # (K, 128)
            if user in VAL_USERS:
                val_X.append(mean_stack);  val_y.append(label);  val_u.append(user)
            else:
                test_X.append(mean_stack); test_y.append(label); test_u.append(user)
        else:
            # Expand by position: pair k-th rep of each modality together
            counts = [len(reps[m][(user, group)]) for m in modality_names]
            n      = min(counts)
            for idx in range(n):
                sample = np.stack([reps[m][(user, group)][idx] for m in modality_names], axis=0)
                train_X.append(sample); train_y.append(label); train_u.append(user)

    def _pack(X, y, u):
        return (np.stack(X).astype(np.float32),
                np.array(y, dtype=np.int32),
                np.array(u))

    return (
        *_pack(train_X, train_y, train_u),
        *_pack(val_X,   val_y,   val_u),
        *_pack(test_X,  test_y,  test_u),
        label_map,
        # Also return mean-pooled full set for embedding extraction
        common_keys,
    )


def build_mean_pooled(reps: dict, modality_names: list, common_keys: list, label_map: dict):
    """Mean-pooled embeddings for ALL keys (for saving fusion_embeddings.npz)."""
    X, y, users, groups = [], [], [], []
    for user, group in common_keys:
        X.append(np.stack([np.mean(reps[m][(user, group)], axis=0) for m in modality_names], axis=0))
        y.append(label_map[group])
        users.append(user)
        groups.append(group)
    return (np.stack(X).astype(np.float32), np.array(y, dtype=np.int32),
            np.array(users), np.array(groups))


# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):
    def __init__(self, X, y, augment=False, num_modalities=5):
        self.X       = torch.from_numpy(X).float()   # (N, K, 128)
        self.y       = torch.from_numpy(y).long()    # (N,)
        self.augment = augment
        self.K       = num_modalities

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x     = self.X[idx].clone()   # (K, 128)
        label = self.y[idx]

        if self.augment:
            # Modality dropout: zero one random modality
            if np.random.rand() < MODALITY_DROP_PROB:
                x[np.random.randint(self.K)] = 0.0
            # Small embedding noise
            x = x + torch.randn_like(x) * EMBED_NOISE_STD

        return x, label


# ============================================================
# MODEL — Cross-Modal Transformer Fusion
# ============================================================

class TransformerFusionLayer(nn.Module):
    """
    Single pre-norm Transformer encoder layer over K modality tokens.
    Returns the updated token tensor and the (B, K, K) attention weight matrix.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, return_weights: bool = False):
        # x: (B, K, d_model)
        normed = self.norm1(x)
        attn_out, weights = self.attn(
            normed, normed, normed,
            need_weights=return_weights,
            average_attn_weights=True,   # average over heads → (B, K, K)
        )
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x, weights


class TransformerFusion(nn.Module):
    """
    Cross-modal Transformer fusion.

    Architecture:
        Input: (B, K, 128)
        ↓ per-modality LayerNorm  (handles encoder scale differences)
        ↓ add learned modality embeddings  (so transformer knows which is which)
        ↓ N_LAYERS × TransformerEncoderLayer (self-attention over K tokens)
        ↓ final LayerNorm
        ↓ mean-pool over K tokens → (B, 128)
        ↓ Linear(128→256)→LN→GELU→Dropout → Linear(256→num_classes)

    Interpretability:
        The (B, K, K) attention matrix from the last layer is averaged over
        queries to give a (B, K) vector — how much each modality was attended
        to (its importance as a "key"). This is the per-sample modality weight
        saved in the output NPZ and plotted in the heatmap.
    """

    def __init__(self, num_modalities: int, embedding_dim: int = 128,
                 num_classes: int = 30, n_heads: int = 4,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.num_modalities = num_modalities
        self.embedding_dim  = embedding_dim

        # Per-modality LayerNorm
        self.modality_norms = nn.ModuleList([
            nn.LayerNorm(embedding_dim) for _ in range(num_modalities)
        ])

        # Learned modality token embeddings (like positional embeddings in ViT)
        self.modality_embed = nn.Embedding(num_modalities, embedding_dim)

        # Transformer encoder layers
        self.layers = nn.ModuleList([
            TransformerFusionLayer(embedding_dim, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)

        # Classifier MLP on mean-pooled output
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def _encode(self, x: torch.Tensor, return_weights: bool = False):
        """
        x: (B, K, 128)
        Returns:
            pooled  : (B, 128) — mean-pooled transformer output
            weights : (B, K)   — modality importance from last layer attention
        """
        B, K, D = x.shape

        # Per-modality norm
        x = torch.stack(
            [self.modality_norms[i](x[:, i, :]) for i in range(K)], dim=1
        )

        # Add modality embeddings
        modal_ids = torch.arange(K, device=x.device)
        x = x + self.modality_embed(modal_ids).unsqueeze(0)  # (B, K, D)

        # Transformer layers
        last_attn = None
        for i, layer in enumerate(self.layers):
            is_last = (i == len(self.layers) - 1)
            x, w = layer(x, return_weights=(return_weights and is_last))
            if is_last:
                last_attn = w

        x = self.final_norm(x)         # (B, K, D)
        pooled = x.mean(dim=1)         # (B, D) — mean pool over modalities

        # Derive modality importance: mean over query dim of last-layer attention
        # (B, K, K) → mean over dim 1 → (B, K): how much each modality was a key
        if return_weights and last_attn is not None:
            modal_weights = last_attn.mean(dim=1)   # (B, K)
        else:
            modal_weights = torch.full((B, K), 1.0 / K, device=x.device)

        return pooled, modal_weights

    def forward(self, x: torch.Tensor):
        pooled, weights = self._encode(x, return_weights=True)
        logits = self.classifier(pooled)
        return logits, weights

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """Returns (pooled_embedding, modal_weights) without grad."""
        return self._encode(x, return_weights=True)


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
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    patience_ctr = 0

    print("\n" + "=" * 60)
    print("TRAINING FUSION TRANSFORMER")
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
# VISUALIZATION
# ============================================================

def plot_attention_by_class(weights, labels, label_map, modality_names, path):
    if not MATPLOTLIB_AVAILABLE:
        print("Skipping attention plot (matplotlib not available)")
        return

    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes  = len(label_map)
    K            = len(modality_names)

    mean_w = np.zeros((num_classes, K))
    for cls in range(num_classes):
        mask = np.array(labels) == cls
        if mask.sum() > 0:
            mean_w[cls] = weights[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(5, K * 1.8), max(8, num_classes * 0.35)))
    im = ax.imshow(mean_w, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(K))
    ax.set_xticklabels(modality_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([idx_to_label[i] for i in range(num_classes)], fontsize=8)
    ax.set_xlabel("Modality", fontsize=12)
    ax.set_ylabel("Utterance class", fontsize=12)
    ax.set_title("Mean cross-modal attention weight per class × modality\n"
                 "(darker = attended to more by other modalities)", fontsize=11)
    for r in range(num_classes):
        for c in range(K):
            ax.text(c, r, f"{mean_w[r, c]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if mean_w[r, c] > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Attention heatmap saved to: {path}")


def print_attention_summary(weights, labels, label_map, modality_names):
    idx_to_label  = {v: k for k, v in label_map.items()}
    label_strings = np.array([idx_to_label[l] for l in labels])
    group_types   = ["sentences", "vowel", "word"]

    print("\nMean cross-modal attention weight per modality:")
    print(f"  {'Modality':<12} {'Overall':>8}", end="")
    for gt in group_types:
        print(f"  {gt:>10}", end="")
    print()

    overall = weights.mean(axis=0)
    for j, name in enumerate(modality_names):
        print(f"  {name:<12} {overall[j]:>8.3f}", end="")
        for gt in group_types:
            mask = np.array([gt in s for s in label_strings])
            w = weights[mask, j].mean() if mask.sum() > 0 else float("nan")
            print(f"  {w:>10.3f}", end="")
        print()


# ============================================================
# MAIN
# ============================================================

def main():
    modality_names = list(NPZ_FILES.keys())

    # ----------------------------------------------------------
    # 1. Load all repetitions from each NPZ
    # ----------------------------------------------------------
    print("Loading all repetitions from each modality...")
    reps = load_all_repetitions(NPZ_FILES)

    # ----------------------------------------------------------
    # 2. Build aligned splits
    # ----------------------------------------------------------
    print("\nBuilding aligned splits...")
    (train_X, train_y, train_u,
     val_X,   val_y,   val_u,
     test_X,  test_y,  test_u,
     label_map, common_keys) = build_aligned_splits(reps, modality_names)

    num_classes    = len(label_map)
    num_modalities = len(modality_names)

    print(f"\n{num_classes} classes | {num_modalities} modalities: {modality_names}")
    print(f"Chance accuracy: 1/{num_classes} = {1/num_classes:.3f}")
    print(f"\nSplit (speaker-independent, train expanded by repetition):")
    print(f"  Train: {len(train_X):4d} samples  (users 1-16 excl 11, per-rep expanded)")
    print(f"  Val:   {len(val_X):4d} samples  (users {VAL_USERS}, mean-pooled)")
    print(f"  Test:  {len(test_X):4d} samples  (users {TEST_USERS}, mean-pooled)")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"\nLabel map saved to {LABEL_MAP_PATH}")

    # ----------------------------------------------------------
    # 3. Datasets and loaders
    # ----------------------------------------------------------
    train_ds = FusionDataset(train_X, train_y, augment=True,  num_modalities=num_modalities)
    val_ds   = FusionDataset(val_X,   val_y,   augment=False, num_modalities=num_modalities)
    test_ds  = FusionDataset(test_X,  test_y,  augment=False, num_modalities=num_modalities)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ----------------------------------------------------------
    # 4. Build model
    # ----------------------------------------------------------
    model = TransformerFusion(
        num_modalities=num_modalities,
        embedding_dim=EMBEDDING_DIM,
        num_classes=num_classes,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTransformer fusion parameters: {total_params:,}")
    print(f"Input: ({num_modalities}, {EMBEDDING_DIM}) → {N_LAYERS}×self-attn → mean-pool → {num_classes} classes")

    # ----------------------------------------------------------
    # 5. Train
    # ----------------------------------------------------------
    model = run_training(model, train_loader, val_loader)

    # ----------------------------------------------------------
    # 6. Test evaluation
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
    # 7. Attention analysis (val + test)
    # ----------------------------------------------------------
    _, val_acc, _, val_labels, val_weights = eval_epoch(model, val_loader, criterion)
    print(f"Val accuracy: {val_acc:.3f}")

    eval_weights = np.concatenate([val_weights, test_weights], axis=0)
    eval_labels  = val_labels + test_labels
    print_attention_summary(eval_weights, eval_labels, label_map, modality_names)
    plot_attention_by_class(eval_weights, eval_labels, label_map, modality_names, ATTN_PLOT_PATH)

    # ----------------------------------------------------------
    # 8. Save fused embeddings (mean-pooled, all aligned samples)
    # ----------------------------------------------------------
    all_X, all_y, all_users, all_groups = build_mean_pooled(
        reps, modality_names, common_keys, label_map
    )
    all_ds     = FusionDataset(all_X, all_y, augment=False, num_modalities=num_modalities)
    all_loader = DataLoader(all_ds, batch_size=64, shuffle=False)

    model.eval()
    fused_embs, all_attn_weights = [], []
    with torch.no_grad():
        for x_batch, _ in all_loader:
            x_batch = x_batch.to(DEVICE)
            emb, w  = model.encode(x_batch)
            fused_embs.append(emb.cpu().numpy())
            all_attn_weights.append(w.cpu().numpy())

    fused_embs       = np.concatenate(fused_embs,       axis=0)  # (N, 128)
    all_attn_weights = np.concatenate(all_attn_weights, axis=0)  # (N, K)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=fused_embs,
        labels=all_y,
        users=all_users,
        groups=all_groups,
        attention_weights=all_attn_weights,
        modality_names=np.array(modality_names),
    )
    print(f"\nFused embeddings saved to {EMBEDDINGS_PATH}")
    print(f"  embeddings shape:        {fused_embs.shape}")
    print(f"  attention_weights shape: {all_attn_weights.shape}")


if __name__ == "__main__":
    main()

"""
model stats

Final validation accuracy: 0.492

Test accuracy (users ['19', '20']): 0.550
accuracy | f1-score: 0.55, support: 60
macro avg | Precision: 0.66, Recall: 0.55, f1-score: 0.56, support: 60
weighted avg | Precision: 0.66, Recall: 0.55, f1-score: 0.56, support: 60
"""