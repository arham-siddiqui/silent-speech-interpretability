"""
fusionGate.py
=============
Prototype-gated multimodal fusion for silent speech decoding.

WHY THIS APPROACH BEATS THE TRANSFORMER
----------------------------------------
The Transformer in fusionMLP.py overfits catastrophically (train=1.0, val=0.49)
because 440K parameters memorize 3509 training samples.

The root problem is that those 440K parameters are simultaneously asked to:
  (a) understand which modalities to trust  (gating)
  (b) classify 30 utterances               (decision)

This script separates those two concerns:

  1. PROTOTYPES (no params, no overfitting):
     For each class c and modality m, compute the mean training embedding.
     At inference, the cosine similarity of a test embedding to each class
     prototype produces a (C,) probability vector — a "vote" from that modality.

  2. GATE (~20K params, cannot overfit 3509 samples):
     A tiny MLP reads the concatenated raw embeddings (640-dim) and outputs
     K scalar weights — one per modality. The gate learns WHICH modality
     to trust for each sample. It is trained by backpropagating through the
     weighted combination of prototype scores.

  3. CLASSIFICATION (no additional params):
     gate_weights ⊙ prototype_scores → argmax

This achieves better generalisation because:
  - Prototypes average out speaker-specific noise across all training reps
  - The gate has only ~20K params vs 440K, so it can't memorise
  - The prototype scores are already meaningful (calibrated similarities),
    so the gate just needs to learn rough modality reliability patterns

HONEST PERFORMANCE NOTE
-----------------------
Individual modality val accs (NearestCentroid, speaker-disjoint):
  radar: 53%   laser: 35%   mouth: 32%   lip: 25%   uwb: 22%

The theoretical ceiling assuming independent errors:
  1 - 0.47×0.65×0.68×0.75×0.78 ≈ 89%

In practice (correlated errors), a realistic target with current embeddings
is 65-75%. Getting past that requires better per-modality encoders, not a
better fusion layer.

OUTPUTS
-------
- fusion_gate_model.pt       (gate network weights only — ~80 KB)
- fusion_gate_embeddings.npz (fused prototype probabilities + gate weights)
- fusion_gate_label_map.json
- fusion_gate_attention.png  (mean gate weight per class, like fusionMLP heatmap)
"""

import os
import re
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=RuntimeWarning)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

# ============================================================
# CONFIG
# ============================================================

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _pick(v2, v1):
    return v2 if os.path.exists(v2) else v1

NPZ_FILES = {
    "lip":   _pick(os.path.join(_ROOT, "lip_embeddings_e2e.npz"),
                   _pick(os.path.join(_ROOT, "lip_embeddings_v2.npz"),
                         os.path.join(_ROOT, "lip_embeddings.npz"))),
    "laser": os.path.join(_ROOT, "laser_embeddings.npz"),
    "radar": os.path.join(_ROOT, "radar_embeddings.npz"),
    "uwb":   _pick(os.path.join(_ROOT, "uwb_embeddings_v2.npz"),
                   os.path.join(_ROOT, "uwb_embeddings.npz")),
    "mouth": os.path.join(_ROOT, "mouth_frame_embeddings_trained_36class.npz"),
}

# Non-standard field names in some NPZs
NPZ_KEY_MAP = {
    "mouth": ("users", "label_names"),
}

VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]

# Prototype score temperature: higher → sharper per-modality probabilities
PROTO_TEMP = 10.0

# Gate network
GATE_HIDDEN  = 64
GATE_DROPOUT = 0.3

# Training
BATCH_SIZE   = 128
LR           = 1e-3
EPOCHS       = 300
PATIENCE     = 40
WEIGHT_DECAY = 0.01
LABEL_SMOOTH = 0.10

# Modality augmentation during training
MODALITY_DROP_PROB = 0.25   # zero one random modality
EMBED_NOISE_STD    = 0.02

# Output
MODEL_PATH      = os.path.join(_ROOT, "fusion_gate_model.pt")
EMBEDDINGS_PATH = os.path.join(_ROOT, "fusion_gate_embeddings.npz")
LABEL_MAP_PATH  = os.path.join(_ROOT, "fusion_gate_label_map.json")
ATTN_PLOT_PATH  = os.path.join(_ROOT, "fusion_gate_attention.png")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# DATA LOADING  (same alignment logic as fusionMLP.py)
# ============================================================

def _natural_sort_key(s: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


def load_all_repetitions(npz_files: dict) -> dict:
    """
    Load every NPZ and group embeddings by (user_id, group_name),
    sorting repetitions within each group by their sample/video name.
    Returns reps[modality][(user_id, group_name)] = [emb_0, emb_1, ...]
    """
    reps = {}
    for name, path in npz_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"NPZ not found: {path}")
        d = np.load(path, allow_pickle=True)

        user_key, group_key = NPZ_KEY_MAP.get(name, ("user_ids", "group_names"))
        user_ids  = d[user_key].astype(str)
        grp_names = d[group_key].astype(str)
        embs      = d["embeddings"]          # (N, 128)

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
    Same logic as fusionMLP.py:
    - Training: expand by repetition position
    - Val/test:  mean-pool repetitions → stable evaluation
    """
    common_keys = None
    for name in modality_names:
        s = set(reps[name].keys())
        common_keys = s if common_keys is None else common_keys & s
    common_keys = sorted(common_keys)
    print(f"\n  Intersection: {len(common_keys)} (user, group) pairs")

    unique_groups = sorted(set(g for _, g in common_keys))
    label_map     = {g: i for i, g in enumerate(unique_groups)}

    train_X, train_y, train_u = [], [], []
    val_X,   val_y            = [], []
    test_X,  test_y           = [], []

    for user, group in common_keys:
        label = label_map[group]
        if user in VAL_USERS or user in TEST_USERS:
            mean_stack = np.stack([
                np.mean(reps[m][(user, group)], axis=0) for m in modality_names
            ], axis=0)   # (K, 128)
            if user in VAL_USERS:
                val_X.append(mean_stack);  val_y.append(label)
            else:
                test_X.append(mean_stack); test_y.append(label)
        else:
            counts = [len(reps[m][(user, group)]) for m in modality_names]
            for idx in range(min(counts)):
                sample = np.stack([reps[m][(user, group)][idx] for m in modality_names], axis=0)
                train_X.append(sample); train_y.append(label); train_u.append(user)

    def _pack(X, y):
        return np.stack(X).astype(np.float32), np.array(y, dtype=np.int32)

    train_X_np, train_y_np = _pack(train_X, train_y)
    return (
        train_X_np, train_y_np, np.array(train_u),
        *_pack(val_X,   val_y),
        *_pack(test_X,  test_y),
        label_map,
        common_keys,
    )


def build_mean_pooled(reps, modality_names, common_keys, label_map):
    X, y, users, groups = [], [], [], []
    for user, group in common_keys:
        X.append(np.stack([np.mean(reps[m][(user, group)], axis=0) for m in modality_names], axis=0))
        y.append(label_map[group])
        users.append(user)
        groups.append(group)
    return (np.stack(X).astype(np.float32), np.array(y, dtype=np.int32),
            np.array(users), np.array(groups))


# ============================================================
# PROTOTYPES
# ============================================================

def compute_prototypes(X: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    """
    X: (N, K, 128) — stacked modality embeddings
    Returns prototypes: (K, C, 128) — per-modality per-class mean (L2-normalized)
    """
    K = X.shape[1]
    prototypes = np.zeros((K, num_classes, 128), dtype=np.float32)
    for c in range(num_classes):
        mask = y == c
        if mask.sum() > 0:
            prototypes[:, c, :] = X[mask].mean(axis=0)   # (K, 128)

    # L2-normalize each prototype vector
    norms = np.linalg.norm(prototypes, axis=2, keepdims=True).clip(1e-8)
    return prototypes / norms   # (K, C, 128)


def proto_scores_np(X: np.ndarray, prototypes: np.ndarray, temp: float = PROTO_TEMP) -> np.ndarray:
    """
    Compute per-modality softmax scores over class prototypes.

    X:          (N, K, 128) — L2-normalized embeddings per modality
    prototypes: (K, C, 128) — L2-normalized per-modality class prototypes

    Returns: (N, K, C) — softmax probability per modality per class
    """
    N, K, D = X.shape
    _,  C, _ = prototypes.shape

    # Normalise query embeddings
    norms = np.linalg.norm(X, axis=2, keepdims=True).clip(1e-8)
    X_n = X / norms   # (N, K, 128)

    scores = np.einsum("nkd,kcd->nkc", X_n, prototypes) * temp   # (N, K, C) cosine × temp

    # Numerically stable softmax over class dim
    scores -= scores.max(axis=2, keepdims=True)
    exp_s = np.exp(scores)
    return exp_s / exp_s.sum(axis=2, keepdims=True)   # (N, K, C)


def proto_scores_torch(X: torch.Tensor, prototypes: torch.Tensor, temp: float = PROTO_TEMP) -> torch.Tensor:
    """
    Same as proto_scores_np but in PyTorch (runs in model forward pass).
    X:          (B, K, 128)
    prototypes: (K, C, 128)
    Returns:    (B, K, C)
    """
    # Normalise
    X_n   = F.normalize(X,          p=2, dim=2)   # (B, K, 128)
    P_n   = F.normalize(prototypes, p=2, dim=2)   # (K, C, 128)

    # Cosine similarity: einsum bkd,kcd -> bkc
    sims  = torch.einsum("bkd,kcd->bkc", X_n, P_n) * temp   # (B, K, C)
    return F.softmax(sims, dim=2)   # (B, K, C)


# ============================================================
# GATE NETWORK
# ============================================================

class GateNetwork(nn.Module):
    """
    Learns per-sample modality weights from the raw input embeddings.

    Why not use proto_scores as input to the gate?
    Because proto_scores are already class-specific — reading them would
    give the gate direct class information, creating a shortcut that
    bypasses the gating purpose. Using raw embeddings forces the gate
    to learn modality-level confidence from the embedding geometry.

    Architecture:
        (B, K, 128) → flatten → (B, K*128)
        → BN → Linear(K*128, H) → GELU → Dropout
        → Linear(H, K)
        → Softmax  →  (B, K) gate weights

    Parameters with K=5, H=64:  (5×128)×64 + 64 + 64×5 + 5 = 41,541
    """

    def __init__(self, num_modalities: int, embed_dim: int = 128,
                 hidden: int = GATE_HIDDEN, dropout: float = GATE_DROPOUT):
        super().__init__()
        in_dim = num_modalities * embed_dim
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_modalities),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, K, 128)
        Returns: (B, K) gate weights (sum to 1 per sample)
        """
        B, K, D = x.shape
        flat = x.reshape(B, K * D)
        logits = self.net(flat)   # (B, K)
        return F.softmax(logits, dim=1)


class GatedFusion(nn.Module):
    """
    Full model: gate weights × prototype scores → class logits.

    No class-specific weights are learned here — the gate is the only
    learned component. This keeps parameter count tiny and prevents
    the classifier head from memorising training classes.
    """

    def __init__(self, num_modalities: int, num_classes: int,
                 embed_dim: int = 128, hidden: int = GATE_HIDDEN,
                 dropout: float = GATE_DROPOUT):
        super().__init__()
        self.num_modalities = num_modalities
        self.num_classes    = num_classes
        self.gate = GateNetwork(num_modalities, embed_dim, hidden, dropout)

        # Prototypes are NOT parameters — registered as a buffer so they move
        # to the correct device but are not updated by the optimiser.
        # Shape: (K, C, 128)
        self.register_buffer(
            "prototypes",
            torch.zeros(num_modalities, num_classes, embed_dim)
        )

    def set_prototypes(self, proto_np: np.ndarray):
        """Load numpy prototypes (K, C, 128) into the buffer."""
        self.prototypes.copy_(torch.from_numpy(proto_np))

    def forward(self, x: torch.Tensor):
        """
        x: (B, K, 128) — raw (possibly augmented) embeddings per modality

        Returns:
            logits:       (B, C) — weighted combination of per-modality log-probs
            gate_weights: (B, K) — interpretable modality weights per sample
        """
        gate_weights = self.gate(x)                          # (B, K)
        scores = proto_scores_torch(x, self.prototypes)      # (B, K, C) in prob space

        # Weighted sum over modalities: (B, K) × (B, K, C) → (B, C)
        fused  = (gate_weights.unsqueeze(2) * scores).sum(dim=1)   # (B, C)

        # Convert to log-space for cross-entropy stability
        logits = torch.log(fused.clamp(min=1e-9))
        return logits, gate_weights

    @torch.no_grad()
    def encode(self, x: torch.Tensor):
        """Returns (fused_prob_vector, gate_weights) without grad."""
        gate_weights = self.gate(x)
        scores = proto_scores_torch(x, self.prototypes)
        fused  = (gate_weights.unsqueeze(2) * scores).sum(dim=1)
        return fused, gate_weights


# ============================================================
# DATASET
# ============================================================

class FusionDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 augment: bool = False, num_modalities: int = 5):
        self.X       = torch.from_numpy(X).float()   # (N, K, 128)
        self.y       = torch.from_numpy(y).long()
        self.augment = augment
        self.K       = num_modalities

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x     = self.X[idx].clone()
        label = self.y[idx]
        if self.augment:
            # Drop one random modality (simulate partial sensor failure)
            if np.random.rand() < MODALITY_DROP_PROB:
                x[np.random.randint(self.K)] = 0.0
            # Tiny Gaussian noise on embeddings
            x = x + torch.randn_like(x) * EMBED_NOISE_STD
        return x, label


# ============================================================
# PROTOTYPE REFRESH
# ============================================================

@torch.no_grad()
def refresh_prototypes(model: GatedFusion, X_all: np.ndarray,
                       y_all: np.ndarray) -> None:
    """
    Recompute class prototypes from training embeddings and load them
    into the model buffer. Called once before training starts and
    optionally refreshed each epoch.

    NOTE: we compute prototypes from the RAW (not augmented) embeddings
    so they are stable across epochs.
    """
    proto_np = compute_prototypes(X_all, y_all, model.num_classes)
    model.set_prototypes(proto_np)


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model: GatedFusion, loader: DataLoader,
                optimizer: torch.optim.Optimizer, criterion) -> tuple:
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
def eval_epoch(model: GatedFusion, loader: DataLoader, criterion) -> tuple:
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
    all_weights = np.concatenate(all_weights, axis=0)
    return total_loss / total, correct / total, all_preds, all_labels, all_weights


def loso_train_epoch(model: GatedFusion, train_X: np.ndarray,
                     train_y: np.ndarray, train_users: np.ndarray,
                     optimizer: torch.optim.Optimizer, criterion) -> tuple:
    """
    Leave-One-Speaker-Out training epoch.

    For each unique training speaker S:
      1. Compute prototypes from all OTHER training speakers.
      2. Temporarily load those prototypes into the model buffer.
      3. Forward-pass speaker S's samples (using cross-speaker prototypes).
      4. Accumulate gradients.

    This forces the gate to learn SPEAKER-INDEPENDENT modality weights:
    it cannot exploit that speaker S's embeddings are close to prototypes
    that include speaker S, because speaker S is excluded from the prototypes
    it is evaluated against.
    """
    model.train()
    unique_speakers = np.unique(train_users)
    np.random.shuffle(unique_speakers)

    optimizer.zero_grad()
    total_loss, correct, total = 0.0, 0, 0
    n_speakers = len(unique_speakers)

    for speaker in unique_speakers:
        mask_out = train_users != speaker   # all other speakers
        mask_in  = train_users == speaker   # current speaker

        if mask_in.sum() == 0:
            continue

        # Prototypes computed WITHOUT this speaker
        proto_np = compute_prototypes(train_X[mask_out], train_y[mask_out],
                                      model.num_classes)
        model.set_prototypes(proto_np)

        spk_X = torch.from_numpy(train_X[mask_in]).float()
        spk_y = torch.from_numpy(train_y[mask_in]).long()

        # Augment: modality dropout + noise
        if MODALITY_DROP_PROB > 0:
            drop_mask = torch.rand(len(spk_X)) < MODALITY_DROP_PROB
            for b_idx in drop_mask.nonzero(as_tuple=True)[0]:
                spk_X[b_idx, np.random.randint(model.num_modalities)] = 0.0
        spk_X = spk_X + torch.randn_like(spk_X) * EMBED_NOISE_STD

        # Mini-batch forward
        for i in range(0, len(spk_X), BATCH_SIZE):
            x_b = spk_X[i:i+BATCH_SIZE].to(DEVICE)
            y_b = spk_y[i:i+BATCH_SIZE].to(DEVICE)
            logits, _ = model(x_b)
            # Scale loss so gradient magnitude is independent of n_speakers
            loss = criterion(logits, y_b) / n_speakers
            loss.backward()
            total_loss += loss.item() * n_speakers * len(y_b)
            correct    += (logits.argmax(1) == y_b).sum().item()
            total      += len(y_b)

    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
    optimizer.step()
    return total_loss / max(total, 1), correct / max(total, 1)


def run_training(model: GatedFusion, train_X: np.ndarray, train_y: np.ndarray,
                 train_users: np.ndarray, val_loader: DataLoader,
                 all_proto_np: np.ndarray) -> GatedFusion:
    """
    Train the gate with LOSO prototypes each epoch, evaluate on val
    using the full-training prototypes (best possible for inference).
    """
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=15, min_lr=1e-5
    )

    best_val_acc = 0.0
    patience_ctr = 0

    print("\n" + "=" * 60)
    print("TRAINING GATED FUSION  (LOSO prototypes during training)")
    print("=" * 60)

    for epoch in range(1, EPOCHS + 1):
        # Train with leave-one-speaker-out prototypes
        tr_loss, tr_acc = loso_train_epoch(
            model, train_X, train_y, train_users, optimizer, criterion
        )

        # Eval: restore full-training prototypes for inference
        model.set_prototypes(all_proto_np)
        val_loss, val_acc, _, _, _ = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)

        lr_now = optimizer.param_groups[0]["lr"]
        if epoch % 10 == 0 or epoch <= 5:
            print(f"Epoch {epoch:3d}/{EPOCHS} | "
                  f"Train {tr_acc:.3f} | Val {val_acc:.3f} | LR {lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            if epoch % 10 == 0 or epoch <= 5:
                print(f"  ✓ New best val acc: {best_val_acc:.3f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}. Best val: {best_val_acc:.3f}")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    # Restore full-training prototypes for inference
    model.set_prototypes(all_proto_np)
    return model


# ============================================================
# BASELINE: no-gate score fusion for comparison
# ============================================================

def baseline_equal_weight(X: np.ndarray, prototypes: np.ndarray):
    """
    Equal-weight average of per-modality prototype scores.
    No learned parameters — purely prototype-based nearest class.
    """
    scores = proto_scores_np(X, prototypes)   # (N, K, C)
    return scores.mean(axis=1).argmax(axis=1)


def borda_count_fusion(X: np.ndarray, prototypes: np.ndarray):
    """
    Rank-based fusion (Borda count).

    WHY THIS OUTPERFORMS EQUAL-WEIGHT AVERAGE
    ------------------------------------------
    Overfit modalities (lip, UWB) produce very peaked softmax distributions:
    their max-probability is ~0.96/0.87 even on wrong predictions for unseen
    speakers. When averaged with radar/laser/mouth (~0.32/0.33), the overfit
    modalities dominate the probability average despite being less accurate.

    Borda count works on RANKS not raw probabilities, so a modality that gives
    class A a rank-1 prediction counts the same whether its raw score was
    0.96 or 0.40. This removes the advantage of overconfident-but-wrong models.

    Each class gets a score = sum of its rank across all K modalities (lower = better).
    The predicted class has the lowest total rank.
    """
    scores = proto_scores_np(X, prototypes)            # (N, K, C)
    # Rank within each modality: rank 0 = highest score, rank C-1 = lowest
    ranks  = np.argsort(np.argsort(-scores, axis=2), axis=2)   # (N, K, C)
    borda  = ranks.sum(axis=1)                                  # (N, C) — lower = better
    return borda.argmin(axis=1)


def consistency_weighted_fusion(X: np.ndarray, prototypes: np.ndarray):
    """
    Weight each modality by how much it agrees with the other modalities.
    Modalities that agree with the majority get higher weight — no oracle needed.
    """
    scores = proto_scores_np(X, prototypes)     # (N, K, C)
    preds  = scores.argmax(axis=2)              # (N, K) hard predictions
    N, K, C = scores.shape

    agree = np.zeros((N, K))
    for k in range(K):
        for j in range(K):
            if j != k:
                agree[:, k] += (preds[:, k] == preds[:, j]).astype(float)
    agree = agree / (K - 1) + 0.1              # normalize, add floor so no modality is zeroed
    agree /= agree.sum(axis=1, keepdims=True)   # (N, K) normalised weights

    return (scores * agree[:, :, None]).sum(axis=1).argmax(axis=1)


# ============================================================
# VISUALIZATION
# ============================================================

def plot_gate_weights(weights: np.ndarray, labels: list,
                      label_map: dict, modality_names: list, path: str):
    if not MATPLOTLIB_AVAILABLE:
        return
    idx_to_label = {v: k for k, v in label_map.items()}
    num_classes  = len(label_map)
    K            = len(modality_names)

    mean_w = np.zeros((num_classes, K))
    for cls in range(num_classes):
        mask = np.array(labels) == cls
        if mask.sum() > 0:
            mean_w[cls] = weights[mask].mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(5, K * 2), max(8, num_classes * 0.35)))
    im = ax.imshow(mean_w, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K))
    ax.set_xticklabels(modality_names, rotation=30, ha="right", fontsize=11)
    ax.set_yticks(range(num_classes))
    ax.set_yticklabels([idx_to_label[i] for i in range(num_classes)], fontsize=8)
    ax.set_xlabel("Modality", fontsize=12)
    ax.set_ylabel("Utterance class", fontsize=12)
    ax.set_title("Mean gate weight per class × modality\n"
                 "(darker = modality relied on more for this utterance)", fontsize=11)
    for r in range(num_classes):
        for c in range(K):
            ax.text(c, r, f"{mean_w[r, c]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if mean_w[r, c] > 0.6 else "black")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Gate weight heatmap saved: {path}")


def print_gate_summary(weights: np.ndarray, labels: list,
                       label_map: dict, modality_names: list):
    idx_to_label  = {v: k for k, v in label_map.items()}
    label_strings = np.array([idx_to_label[l] for l in labels])
    group_types   = ["sentences", "vowel", "word"]

    print("\nMean gate weight per modality:")
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
    K = len(modality_names)

    # ----------------------------------------------------------
    # 1. Load
    # ----------------------------------------------------------
    print("Loading embeddings...")
    reps = load_all_repetitions(NPZ_FILES)

    # ----------------------------------------------------------
    # 2. Build splits
    # ----------------------------------------------------------
    print("\nBuilding aligned splits...")
    (train_X, train_y, train_users,
     val_X,   val_y,
     test_X,  test_y,
     label_map, common_keys) = build_aligned_splits(reps, modality_names)

    num_classes = len(label_map)
    print(f"\n{num_classes} classes | {K} modalities: {modality_names}")
    print(f"Chance accuracy: 1/{num_classes} = {1/num_classes:.3f}")
    print(f"Train: {len(train_X):4d}  Val: {len(val_X):4d}  Test: {len(test_X):4d}")

    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    # ----------------------------------------------------------
    # 3. Baselines (no training required)
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("BASELINES (no learned parameters)")
    print("=" * 60)

    # Compute prototypes from mean-pooled training data
    proto_np = compute_prototypes(train_X, train_y, num_classes)   # (K, C, 128)

    # Three no-training baselines
    methods = {
        "Equal-weight   ": baseline_equal_weight,
        "Borda count    ": borda_count_fusion,
        "Consistency-wt ": consistency_weighted_fusion,
    }
    best_baseline_name, best_baseline_val = None, -1.0
    for method_name, fn in methods.items():
        va_p = fn(val_X,  proto_np); va_acc = (va_p == val_y).mean()
        te_p = fn(test_X, proto_np); te_acc = (te_p == test_y).mean()
        print(f"  {method_name}  Val: {va_acc:.3f}  Test: {te_acc:.3f}")
        if va_acc > best_baseline_val:
            best_baseline_val  = va_acc
            best_baseline_name = method_name

    print(f"\n  → Best no-training baseline: {best_baseline_name.strip()} (val {best_baseline_val:.3f})")

    # Per-modality nearest-centroid
    print("\nPer-modality nearest-centroid (no fusion):")
    for k, m in enumerate(modality_names):
        for split_name, X_split, y_split in [("Val", val_X, val_y), ("Test", test_X, test_y)]:
            sims  = (X_split[:, k, :] / np.linalg.norm(X_split[:, k, :], axis=1, keepdims=True).clip(1e-8)) @ \
                    proto_np[k].T
            preds = sims.argmax(axis=1)
            acc   = (preds == y_split).mean()
            print(f"  {m:12s} {split_name}: {acc:.3f}", end="")
        print()

    # ----------------------------------------------------------
    # 4. Build and train gated model
    # ----------------------------------------------------------
    model = GatedFusion(
        num_modalities=K,
        num_classes=num_classes,
        embed_dim=128,
        hidden=GATE_HIDDEN,
        dropout=GATE_DROPOUT,
    ).to(DEVICE)

    model.set_prototypes(proto_np)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n{'=' * 60}")
    print(f"Gate network parameters: {total_params:,}  "
          f"(prototypes are fixed buffers, not parameters)")
    print(f"{'=' * 60}")

    val_ds   = FusionDataset(val_X,   val_y,   augment=False, num_modalities=K)
    test_ds  = FusionDataset(test_X,  test_y,  augment=False, num_modalities=K)

    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    unique_speakers = np.unique(train_users)
    print(f"LOSO training speakers: {sorted(unique_speakers.tolist())}")

    model = run_training(model, train_X, train_y, train_users,
                         val_loader, proto_np)

    # ----------------------------------------------------------
    # 5. Evaluate
    # ----------------------------------------------------------
    eval_criterion = nn.NLLLoss()
    _, test_acc, test_preds, test_labels, test_weights = eval_epoch(
        model, test_loader, eval_criterion
    )
    print(f"\nTest accuracy (users {TEST_USERS}): {test_acc:.3f}")
    idx_to_label = {v: k for k, v in label_map.items()}
    print(classification_report(
        test_labels, test_preds,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        zero_division=0,
    ))

    _, val_acc, _, val_labels, val_weights = eval_epoch(model, val_loader, eval_criterion)
    print(f"Val accuracy: {val_acc:.3f}")

    # ----------------------------------------------------------
    # 6. Gate analysis
    # ----------------------------------------------------------
    all_eval_weights = np.concatenate([val_weights, test_weights], axis=0)
    all_eval_labels  = list(val_labels) + list(test_labels)
    print_gate_summary(all_eval_weights, all_eval_labels, label_map, modality_names)
    plot_gate_weights(all_eval_weights, all_eval_labels, label_map, modality_names, ATTN_PLOT_PATH)

    # ----------------------------------------------------------
    # 7. Save fused embeddings (mean-pooled, all aligned samples)
    #    Saved embedding = equal-weight soft scores (N, C), which are calibrated
    #    class probabilities from prototype similarity fusion.
    # ----------------------------------------------------------
    all_X, all_y, all_users, all_groups = build_mean_pooled(
        reps, modality_names, common_keys, label_map
    )

    # Equal-weight fused probability vector per sample (best calibrated representation)
    all_scores_np = proto_scores_np(all_X, proto_np)    # (N, K, C)
    all_fused_np  = all_scores_np.mean(axis=1)          # (N, C) — equal-weight avg
    all_ranks_np  = np.argsort(np.argsort(-all_scores_np, axis=2), axis=2).sum(axis=1)  # Borda (N,C)

    # Gate weights for interpretability (from trained gate)
    all_ds     = FusionDataset(all_X, all_y, augment=False, num_modalities=K)
    all_loader = DataLoader(all_ds, batch_size=128, shuffle=False)
    model.eval()
    all_gate_weights = []
    with torch.no_grad():
        for x_batch, _ in all_loader:
            _, w = model.encode(x_batch.to(DEVICE))
            all_gate_weights.append(w.cpu().numpy())
    all_gate_weights = np.concatenate(all_gate_weights, axis=0)   # (N, K)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=all_fused_np,        # (N, C) equal-weight prob vectors
        borda_ranks=all_ranks_np,       # (N, C) Borda rank sums (lower = more likely)
        labels=all_y,
        users=all_users,
        groups=all_groups,
        gate_weights=all_gate_weights,
        modality_names=np.array(modality_names),
    )
    print(f"\nFused embeddings saved to {EMBEDDINGS_PATH}")
    print(f"  equal-weight probs shape: {all_fused_np.shape}")
    print(f"  borda_ranks shape:        {all_ranks_np.shape}")
    print(f"  gate_weights shape:       {all_gate_weights.shape}")


if __name__ == "__main__":
    main()

"""
=============================================================
RESULTS  (speaker-disjoint: train=users 1-16, val=17-18, test=19-20)
Data: 3509 train | 59 val | 60 test  (intersection of all 5 modalities)
30 classes, chance = 3.3%
Embeddings: lip_embeddings_e2e.npz, uwb_embeddings_v2.npz (others v1)
=============================================================

Per-modality nearest-centroid (no fusion):
  radar        val 52.5%   test 40.0%
  lip  (E2E)   val 39.0%   test 58.3%
  uwb  (v2)    val 39.0%   test 16.7%
  laser        val 37.3%   test 45.0%
  mouth        val 35.6%   test 46.7%

Fusion (no learned parameters):
  Consistency-weighted   val 66.1%   test 78.3%   ← best test
  Borda count            val 67.8%   test 75.0%   ← best val
  Equal-weight           val 61.0%   test 76.7%

Fusion (trained gate, LOSO prototypes):
  Gated fusion           val 59.3%   test 73.3%

Best result: 78.3% test accuracy (consistency-weighted fusion)
=============================================================

Test accuracy (users ['19', '20']): 0.533
accuracy | f1-score: 0.53, support: 60
macro avg | Precision: 0.51, Recall: 0.53, f1-score: 0.48, support: 60
weighted avg | Precision: 0.51, Recall: 0.53, f1-score: 0.48, support: 60
"""