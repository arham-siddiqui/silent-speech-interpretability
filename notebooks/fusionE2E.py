"""
fusionE2E.py
============
End-to-end joint fine-tuning of the lip encoder + fusion head.

WHY END-TO-END MATTERS
-----------------------
In the two-stage pipeline (train encoders → freeze → train fusion), the
lip encoder optimised for its own per-modality classification loss. That
loss does not know which lip features are most useful GIVEN the other four
modalities. Joint training lets the fusion loss propagate gradients back
through the lip encoder, so it learns to produce embeddings that complement
what radar/laser/mouth/UWB already capture.

STRATEGY
--------
1. Load pre-trained lip encoder (liplandmarkLSTM_v2.pt) — kept TRAINABLE.
2. Load pre-computed, FROZEN embeddings for the other four modalities
   (laser, radar, UWB, mouth) from their NPZ files.
3. For each training sample, look up the matching frozen embeddings by
   (user_id, group_name, video/sample index), run the raw lip data through
   the live lip encoder, then fuse all five embeddings with the prototype-
   based gate and compute the classification loss.
4. Backprop updates ONLY the lip encoder + the prototype gate weights.
   The other four encoders stay frozen — their NPZ embeddings are used as-is.

ALIGNMENT
---------
Lip raw data is matched to NPZ embeddings via:
  (user_id, group_name, video_index)
where video_index is the ordinal within that group (0, 1, 2, …), matching
the repetition-expansion used in fusionGate.py.

FUSION HEAD
-----------
Same prototype-based gate as fusionGate.py: prototypes are recomputed from
training embeddings at the start of each epoch (LOSO — leave-one-speaker-out).
The gate MLP learns per-sample modality weights; all classification is done
via cosine distance to prototypes (no linear classification head → less overfit).

OUTPUTS
-------
- fusion_e2e_model.pt         (lip encoder + gate weights)
- lip_embeddings_e2e.npz      (re-extracted lip embeddings after fine-tuning)
- fusion_e2e_embeddings.npz   (fused embeddings for all aligned samples)
- fusion_e2e_label_map.json
"""

import os, re, glob, json, warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from collections import defaultdict
from sklearn.metrics import classification_report

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ============================================================
# PATHS
# ============================================================

ROOT    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIP_RAW = os.path.join(ROOT, "src/data/RVTALL/Processed_cut_data/kinect_processed/")

# NPZ embeddings for the FOUR frozen modalities
def _pick(v2, v1):
    return v2 if os.path.exists(v2) else v1

FROZEN_NPZ = {
    "laser": os.path.join(ROOT, "laser_embeddings.npz"),
    "radar": os.path.join(ROOT, "radar_embeddings.npz"),
    "uwb":   _pick(os.path.join(ROOT, "uwb_embeddings_v2.npz"),
                   os.path.join(ROOT, "uwb_embeddings.npz")),
    "mouth": os.path.join(ROOT, "mouth_frame_embeddings_trained_36class.npz"),
}
FROZEN_KEY_MAP = {"mouth": ("users", "label_names")}   # non-standard field names

# Lip encoder weights (from liplandmarkLSTM_v2.py, or fall back to v1)
LIP_MODEL_V2 = os.path.join(ROOT, "lip_lstm_model_v2.pt")
LIP_MODEL_V1 = os.path.join(ROOT, "lip_lstm_model.pt")

MODEL_OUT_PATH    = os.path.join(ROOT, "fusion_e2e_model.pt")
LIP_EMB_OUT_PATH  = os.path.join(ROOT, "lip_embeddings_e2e.npz")
FUSED_EMB_PATH    = os.path.join(ROOT, "fusion_e2e_embeddings.npz")
LABEL_MAP_PATH    = os.path.join(ROOT, "fusion_e2e_label_map.json")

VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]

# ============================================================
# HYPERPARAMETERS
# ============================================================

EMBEDDING_DIM   = 128
LIP_INPUT_SIZE  = 80      # landmarks (40) + velocity (40)
LIP_HIDDEN      = 256
LIP_LAYERS      = 2
LIP_DROPOUT     = 0.3

GATE_HIDDEN     = 64
GATE_DROPOUT    = 0.3

PROTO_TEMP      = 10.0    # temperature for prototype softmax scores

LIP_LR          = 3e-5    # low LR for fine-tuning (was 3e-4 for from-scratch)
GATE_LR         = 1e-3
WEIGHT_DECAY    = 1e-4
EPOCHS          = 100
PATIENCE        = 30
BATCH_SIZE      = 32

MODALITY_DROP_PROB = 0.25
EMBED_NOISE_STD    = 0.02

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

# ============================================================
# LIP DATA UTILITIES  (same as liplandmarkLSTM_v2.py)
# ============================================================

LIP_START, LIP_END = 48, 68

def list_sorted_npy(directory):
    files = glob.glob(os.path.join(directory, "*.npy"))
    def nk(p):
        ns = re.findall(r"\d+", os.path.basename(p))
        return [int(n) for n in ns] if ns else [os.path.basename(p)]
    return sorted(files, key=nk)

def normalize_landmarks(lm):
    lm = np.asarray(lm, dtype=np.float32)
    c  = lm.mean(axis=0); lm -= c
    s  = np.max(np.linalg.norm(lm, axis=1)) + 1e-8
    return lm / s

def load_lip_seq(landmarkers_dir):
    frames = []
    for f in list_sorted_npy(landmarkers_dir):
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[0] < 68 or arr.shape[1] < 2:
            continue
        lip = arr[LIP_START:LIP_END, :2]
        frames.append(normalize_landmarks(lip).flatten())
    if len(frames) < 5:
        return None
    return np.asarray(frames, dtype=np.float32)

def compute_velocity(seq):
    return np.gradient(seq, axis=0).astype(np.float32)

def _natural_sort_key(s):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ============================================================
# LOAD FROZEN MODALITY EMBEDDINGS
# ============================================================

def load_frozen_embeddings() -> dict:
    """
    Load each frozen-modality NPZ into a dict:
      frozen[modality][(user_id, group_name)] = [emb_rep0, emb_rep1, …]
    Repetitions are sorted by their sample/video name (natural order).
    """
    frozen = {}
    for name, path in FROZEN_NPZ.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Frozen NPZ not found: {path}")
        d = np.load(path, allow_pickle=True)
        uk, gk = FROZEN_KEY_MAP.get(name, ("user_ids", "group_names"))
        users  = d[uk].astype(str)
        groups = d[gk].astype(str)
        embs   = d["embeddings"]
        sort_vals = None
        for k in ["sample_names", "video_names"]:
            if k in d.files:
                sort_vals = d[k].astype(str)
                break

        pool = defaultdict(list)
        for i in range(len(embs)):
            key = (users[i], groups[i])
            sv  = sort_vals[i] if sort_vals is not None else str(i)
            pool[key].append((sv, embs[i]))

        frozen[name] = {
            key: [e for _, e in sorted(items, key=lambda x: _natural_sort_key(x[0]))]
            for key, items in pool.items()
        }
        n_pairs = len(frozen[name])
        n_total = sum(len(v) for v in frozen[name].values())
        print(f"  {name:12s}: {n_pairs} pairs, {n_total} total reps")

    return frozen


# ============================================================
# BUILD LIP SAMPLE LIST WITH VIDEO INDEX
# ============================================================

def build_lip_samples(raw_root):
    """
    Walk kinect_processed/ and return samples sorted by
    (user_id, group_name, video_index) so they align with NPZ ordering.
    Each dict has: user_id, group_name, video_name, lm_dir, video_idx
    """
    samples = []
    user_dirs = sorted(
        [d for d in glob.glob(os.path.join(raw_root, "*")) if os.path.isdir(d)],
        key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])
        if re.findall(r"\d+", os.path.basename(p)) else os.path.basename(p)
    )
    for user_dir in user_dirs:
        user_id = os.path.basename(user_dir)
        for group_prefix in ["sentences", "vowel", "word"]:
            for group_dir in sorted(glob.glob(os.path.join(user_dir, f"{group_prefix}*"))):
                group_name = os.path.basename(group_dir)
                videos_dir = os.path.join(group_dir, "videos")
                if not os.path.isdir(videos_dir):
                    continue
                vid_glob  = "video_[0-9]*" if user_id == "1" else "video_proc_*"
                vid_dirs  = sorted(glob.glob(os.path.join(videos_dir, vid_glob)),
                                   key=lambda p: _natural_sort_key(os.path.basename(p)))
                for idx, vdir in enumerate(vid_dirs):
                    lm_dir = os.path.join(vdir, "landmarkers_cv")
                    if not os.path.isdir(lm_dir):
                        continue
                    samples.append({
                        "user_id":   user_id,
                        "group_name": group_name,
                        "video_name": os.path.basename(vdir),
                        "video_idx":  idx,
                        "lm_dir":    lm_dir,
                        "label_str": group_name,
                    })
    print(f"Found {len(samples)} lip raw samples.")
    return samples


# ============================================================
# ALIGN LIP SAMPLES WITH FROZEN MODALITY REPETITIONS
# ============================================================

def build_aligned_dataset(lip_samples, frozen, label_map):
    """
    For each lip sample (user, group, video_idx), check whether that
    repetition index exists in ALL frozen modalities.
    Returns list of aligned records:
      { lm_dir, frozen_embs: {mod: (128,)}, label, user_id, group_name }
    """
    frozen_modalities = list(frozen.keys())
    aligned = []
    skipped = 0

    for s in lip_samples:
        key       = (s["user_id"], s["group_name"])
        vid_idx   = s["video_idx"]
        label     = label_map.get(s["label_str"])
        if label is None:
            skipped += 1
            continue

        # Check all frozen modalities have this repetition
        ok = True
        frozen_embs = {}
        for m in frozen_modalities:
            reps = frozen[m].get(key)
            if reps is None or vid_idx >= len(reps):
                ok = False
                break
            frozen_embs[m] = reps[vid_idx].astype(np.float32)

        if not ok:
            skipped += 1
            continue

        aligned.append({
            "lm_dir":      s["lm_dir"],
            "frozen_embs": frozen_embs,   # {mod_name: (128,)}
            "label":       label,
            "user_id":     s["user_id"],
            "group_name":  s["group_name"],
            "video_idx":   vid_idx,
        })

    print(f"Aligned: {len(aligned)} samples ({skipped} skipped — no frozen match)")
    return aligned


# ============================================================
# DATASET
# ============================================================

class E2EDataset(Dataset):
    def __init__(self, records, frozen_modalities, augment=False):
        self.frozen_modalities = frozen_modalities
        self.augment = augment
        self.items   = []
        skipped = 0

        for r in records:
            seq = load_lip_seq(r["lm_dir"])
            if seq is None:
                skipped += 1
                continue
            vel  = compute_velocity(seq)
            feat = np.concatenate([seq, vel], axis=1).astype(np.float32)  # (T, 80)
            # Stack frozen embeddings in fixed modality order
            frozens = np.stack([r["frozen_embs"][m] for m in frozen_modalities], axis=0)  # (K_frozen, 128)
            self.items.append((feat, frozens, r["label"], r["user_id"], r["group_name"]))

        print(f"  Loaded {len(self.items)} E2E samples ({skipped} skipped lip-seq errors).")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        seq, frozens, label, user_id, group_name = self.items[idx]

        if self.augment:
            T = len(seq)
            if T > 10:
                keep  = int(np.random.uniform(0.80, 1.0) * T)
                start = np.random.randint(0, T - keep + 1)
                seq   = seq[start:start + keep]
            seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)

            # Frozen embedding noise
            frozens = frozens + np.random.normal(0, EMBED_NOISE_STD, frozens.shape).astype(np.float32)

            # Modality dropout on frozen embeddings
            if np.random.rand() < MODALITY_DROP_PROB:
                frozens[np.random.randint(len(self.frozen_modalities))] = 0.0

        return (torch.from_numpy(seq),
                torch.from_numpy(frozens),
                label, user_id, group_name)


def collate_fn(batch):
    seqs, frozens, labels, uids, gnames = zip(*batch)
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded  = pad_sequence(seqs, batch_first=True)   # (B, T_max, 80)
    frozens = torch.stack(frozens)                   # (B, K_frozen, 128)
    labels  = torch.tensor(labels, dtype=torch.long)
    return padded, lengths, frozens, labels, list(uids), list(gnames)


# ============================================================
# LIP ENCODER  (same arch as liplandmarkLSTM_v2.py, minus speaker head)
# ============================================================

class GradRevFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, a):
        ctx.save_for_backward(a); return x.clone()
    @staticmethod
    def backward(ctx, g):
        a, = ctx.saved_tensors; return -a * g, None

def grad_rev(x, alpha=1.0):
    return GradRevFn.apply(x, torch.tensor(alpha, dtype=x.dtype, device=x.device))


class LipEncoderE2E(nn.Module):
    """BiLSTM with temporal attention — fine-tuneable lip encoder."""
    def __init__(self, input_size=80, hidden_size=256, num_layers=2,
                 embedding_dim=128, dropout=0.3):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True, dropout=dropout if num_layers>1 else 0.0)
        self.attn_proj  = nn.Linear(hidden_size * 2, 1, bias=False)
        self.dropout    = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(nn.Linear(hidden_size * 2, embedding_dim),
                                         nn.LayerNorm(embedding_dim))

    def forward(self, x, lengths):
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True)   # (B, T, 512)
        scores = self.attn_proj(out).squeeze(-1)              # (B, T)
        max_T  = out.size(1)
        mask   = torch.arange(max_T, device=out.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        w      = F.softmax(scores, dim=1).unsqueeze(-1)
        pooled = (out * w).sum(dim=1)                         # (B, 512)
        raw    = self.embed_proj(self.dropout(pooled))        # (B, 128)
        return F.normalize(raw, p=2, dim=1)


# ============================================================
# GATE NETWORK  (same as fusionGate.py)
# ============================================================

class GateNetwork(nn.Module):
    def __init__(self, num_modalities, embed_dim=128, hidden=64, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(num_modalities * embed_dim),
            nn.Linear(num_modalities * embed_dim, hidden),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_modalities),
        )
    def forward(self, x):
        B, K, D = x.shape
        return F.softmax(self.net(x.reshape(B, K*D)), dim=1)   # (B, K)


# ============================================================
# PROTOTYPE UTILITIES
# ============================================================

def compute_prototypes(X: np.ndarray, y: np.ndarray, num_classes: int) -> np.ndarray:
    """X: (N, K, 128), returns (K, C, 128) L2-normed prototypes."""
    K = X.shape[1]
    P = np.zeros((K, num_classes, 128), dtype=np.float32)
    for c in range(num_classes):
        mask = y == c
        if mask.sum() > 0:
            P[:, c, :] = X[mask].mean(axis=0)
    norms = np.linalg.norm(P, axis=2, keepdims=True).clip(1e-8)
    return P / norms


def proto_scores_torch(X: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    """X: (B, K, 128), prototypes: (K, C, 128) → (B, K, C) softmax."""
    X_n = F.normalize(X, p=2, dim=2)
    P_n = F.normalize(prototypes, p=2, dim=2)
    sims = torch.einsum("bkd,kcd->bkc", X_n, P_n) * PROTO_TEMP
    return F.softmax(sims, dim=2)


# ============================================================
# FULL E2E MODEL WRAPPER
# ============================================================

class E2EFusionModel(nn.Module):
    """
    Trainable lip encoder + frozen-embedding gate + prototype fusion.
    Only the lip encoder and gate network have learnable parameters.
    """
    def __init__(self, num_modalities_total, num_classes,
                 num_frozen, embed_dim=128):
        super().__init__()
        self.K = num_modalities_total
        self.lip_enc = LipEncoderE2E(
            input_size=LIP_INPUT_SIZE, hidden_size=LIP_HIDDEN,
            num_layers=LIP_LAYERS, embedding_dim=embed_dim, dropout=LIP_DROPOUT
        )
        self.gate = GateNetwork(num_modalities_total, embed_dim, GATE_HIDDEN, GATE_DROPOUT)
        self.register_buffer("prototypes",
                             torch.zeros(num_modalities_total, num_classes, embed_dim))

    def set_prototypes(self, proto_np: np.ndarray):
        self.prototypes.copy_(torch.from_numpy(proto_np))

    def forward(self, lip_seq, lip_lengths, frozen_embs):
        """
        lip_seq:     (B, T, 80)
        lip_lengths: (B,)
        frozen_embs: (B, K_frozen, 128) — already normalised

        Returns:
            logits:       (B, C)
            gate_weights: (B, K)
        """
        lip_emb = self.lip_enc(lip_seq, lip_lengths)          # (B, 128)
        # Prepend lip embedding to frozen embeddings
        all_embs = torch.cat([lip_emb.unsqueeze(1), frozen_embs], dim=1)  # (B, K, 128)

        gate_w  = self.gate(all_embs)                          # (B, K)
        scores  = proto_scores_torch(all_embs, self.prototypes)  # (B, K, C)
        fused   = (gate_w.unsqueeze(2) * scores).sum(dim=1)   # (B, C)
        logits  = torch.log(fused.clamp(min=1e-9))
        return logits, gate_w, all_embs


# ============================================================
# LOSO PROTOTYPE COMPUTATION  (speaker-independent training)
# ============================================================

def build_train_embeddings_from_records(model, records, frozen_modalities, device):
    """
    Run the CURRENT lip encoder on all training records to get fresh lip
    embeddings, then stack with frozen embeddings to form (N, K, 128).
    Used for LOSO prototype computation each epoch.
    """
    model.eval()
    all_X, all_y, all_u = [], [], []
    with torch.no_grad():
        for r in records:
            seq = load_lip_seq(r["lm_dir"])
            if seq is None:
                continue
            vel  = compute_velocity(seq)
            feat = np.concatenate([seq, vel], axis=1)
            x    = torch.from_numpy(feat).unsqueeze(0).to(device)
            L    = torch.tensor([len(feat)], dtype=torch.long).to(device)
            lip_emb = model.lip_enc(x, L).squeeze(0).cpu().numpy()   # (128,)
            frozen  = np.stack([r["frozen_embs"][m] for m in frozen_modalities])  # (K_fr, 128)
            all_emb = np.concatenate([lip_emb[None], frozen], axis=0)  # (K, 128)
            all_X.append(all_emb)
            all_y.append(r["label"])
            all_u.append(r["user_id"])
    return np.stack(all_X), np.array(all_y), np.array(all_u)


def compute_loso_prototypes(all_X, all_y, all_u, num_classes):
    """
    For each unique training speaker, compute prototypes from all OTHER
    speakers, then return a SINGLE prototype set averaged across speaker
    ablations. This is the LOSO cross-speaker prototype estimate used for
    training the gate.
    """
    speakers = np.unique(all_u)
    proto_sum = np.zeros((all_X.shape[1], num_classes, 128), dtype=np.float64)
    count = 0
    for spk in speakers:
        mask_out = all_u != spk
        if mask_out.sum() == 0:
            continue
        P = compute_prototypes(all_X[mask_out], all_y[mask_out], num_classes)
        proto_sum += P
        count += 1
    return (proto_sum / max(count, 1)).astype(np.float32)


# ============================================================
# TRAINING
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for lip_pad, lip_len, frozens, labels, uids, gnames in loader:
        lip_pad  = lip_pad.to(DEVICE)
        lip_len  = lip_len.to(DEVICE)
        frozens  = frozens.to(DEVICE)
        labels   = labels.to(DEVICE)

        optimizer.zero_grad()
        logits, _, _ = model(lip_pad, lip_len, frozens)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_one_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_gate_w = [], [], []
    for lip_pad, lip_len, frozens, labels, uids, gnames in loader:
        lip_pad  = lip_pad.to(DEVICE)
        lip_len  = lip_len.to(DEVICE)
        frozens  = frozens.to(DEVICE)
        labels   = labels.to(DEVICE)
        logits, gate_w, _ = model(lip_pad, lip_len, frozens)
        loss = criterion(logits, labels)
        total_loss += loss.item() * len(labels)
        correct    += (logits.argmax(1) == labels).sum().item()
        total      += len(labels)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_gate_w.append(gate_w.cpu().numpy())
    all_gate_w = np.concatenate(all_gate_w, axis=0)
    return total_loss / total, correct / total, all_preds, all_labels, all_gate_w


def run_training(model, train_records, frozen_modalities, train_loader, val_loader,
                 num_classes, label_map):
    criterion = nn.NLLLoss()

    # Separate learning rates: low for lip encoder (fine-tune), higher for gate
    param_groups = [
        {"params": model.lip_enc.parameters(), "lr": LIP_LR},
        {"params": model.gate.parameters(),    "lr": GATE_LR},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=12, min_lr=1e-6
    )

    best_val_acc, patience_ctr = 0.0, 0
    idx_to_label = {v: k for k, v in label_map.items()}

    print("\n" + "=" * 70)
    print("END-TO-END FINE-TUNING  (lip encoder lr={:.1e}, gate lr={:.1e})".format(LIP_LR, GATE_LR))
    print("=" * 70)

    for epoch in range(1, EPOCHS + 1):
        # Recompute LOSO prototypes from current lip encoder
        all_X, all_y, all_u = build_train_embeddings_from_records(
            model, train_records, frozen_modalities, DEVICE
        )
        loso_proto = compute_loso_prototypes(all_X, all_y, all_u, num_classes)
        model.set_prototypes(loso_proto)

        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion)

        # Val: use all-training prototypes (best for inference)
        all_proto = compute_prototypes(all_X, all_y, num_classes)
        model.set_prototypes(all_proto)
        val_loss, val_acc, _, _, _ = eval_one_epoch(model, val_loader, criterion)

        scheduler.step(val_acc)
        lr_lip  = optimizer.param_groups[0]["lr"]
        lr_gate = optimizer.param_groups[1]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train {tr_acc:.3f} | Val {val_acc:.3f} | "
              f"lr_lip={lr_lip:.1e} lr_gate={lr_gate:.1e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_OUT_PATH)
            print(f"  ✓ Best val: {best_val_acc:.3f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(MODEL_OUT_PATH, map_location=DEVICE))
    # Restore all-training prototypes for final evaluation
    all_X, all_y, all_u = build_train_embeddings_from_records(
        model, train_records, frozen_modalities, DEVICE
    )
    all_proto = compute_prototypes(all_X, all_y, num_classes)
    model.set_prototypes(all_proto)
    return model


# ============================================================
# MAIN
# ============================================================

def main():
    frozen_modalities = list(FROZEN_NPZ.keys())   # laser, radar, uwb, mouth

    # -------------------------------------------------------
    # 1. Load frozen embeddings
    # -------------------------------------------------------
    print("Loading frozen modality embeddings...")
    frozen = load_frozen_embeddings()

    # -------------------------------------------------------
    # 2. Build lip sample list
    # -------------------------------------------------------
    print("\nBuilding lip sample list...")
    lip_samples = build_lip_samples(LIP_RAW)

    # -------------------------------------------------------
    # 3. Build label map from ALL frozen modality intersections
    # -------------------------------------------------------
    common_keys = set(frozen[frozen_modalities[0]].keys())
    for m in frozen_modalities:
        common_keys &= set(frozen[m].keys())
    all_groups  = sorted(set(g for _, g in common_keys))
    label_map   = {g: i for i, g in enumerate(all_groups)}
    num_classes = len(label_map)
    print(f"\n{num_classes} classes | {len(frozen_modalities)} frozen modalities + lip = "
          f"{len(frozen_modalities)+1} total")
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    # -------------------------------------------------------
    # 4. Align lip samples with frozen embeddings
    # -------------------------------------------------------
    print("\nAligning lip samples with frozen embeddings...")
    aligned = build_aligned_dataset(lip_samples, frozen, label_map)

    train_records = [r for r in aligned if r["user_id"] not in VAL_USERS + TEST_USERS]
    val_records   = [r for r in aligned if r["user_id"] in VAL_USERS]
    test_records  = [r for r in aligned if r["user_id"] in TEST_USERS]
    print(f"Train: {len(train_records)} | Val: {len(val_records)} | Test: {len(test_records)}")

    # -------------------------------------------------------
    # 5. Datasets and loaders
    # -------------------------------------------------------
    print("\nBuilding E2E datasets...")
    train_ds = E2EDataset(train_records, frozen_modalities, augment=True)
    val_ds   = E2EDataset(val_records,   frozen_modalities, augment=False)
    test_ds  = E2EDataset(test_records,  frozen_modalities, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # -------------------------------------------------------
    # 6. Build model — load pre-trained lip weights if available
    # -------------------------------------------------------
    K_total = len(frozen_modalities) + 1   # frozen mods + lip
    model   = E2EFusionModel(
        num_modalities_total=K_total,
        num_classes=num_classes,
        num_frozen=len(frozen_modalities),
    ).to(DEVICE)

    lip_weights_path = LIP_MODEL_V2 if os.path.exists(LIP_MODEL_V2) else LIP_MODEL_V1
    if os.path.exists(lip_weights_path):
        print(f"\nLoading pre-trained lip weights from {os.path.basename(lip_weights_path)}")
        saved = torch.load(lip_weights_path, map_location=DEVICE)
        # Load only keys that match the encoder (strip speaker head, classifier, etc.)
        enc_state = {k.replace("lip_enc.", ""): v for k, v in saved.items()
                     if k.startswith("lip_enc.")}
        # Fall back: load compatible keys directly into lip_enc
        model_keys = set(model.lip_enc.state_dict().keys())
        # Try loading from the v2 model (has same arch as LipEncoderE2E)
        compatible = {}
        for k, v in saved.items():
            # v2 model keys: lstm.*, attn_proj.*, input_norm.*, embed_proj.*
            # These map directly to LipEncoderE2E fields
            if k in model.lip_enc.state_dict() and model.lip_enc.state_dict()[k].shape == v.shape:
                compatible[k] = v
        if compatible:
            model.lip_enc.load_state_dict(compatible, strict=False)
            print(f"  Loaded {len(compatible)}/{len(model.lip_enc.state_dict())} lip encoder keys.")
        else:
            print("  Could not map pre-trained weights (architecture mismatch) — starting fresh.")
    else:
        print("\nNo pre-trained lip weights found — training lip encoder from scratch.")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {total_params:,}  "
          f"(lip encoder + gate; frozen embeddings have no params)")

    # -------------------------------------------------------
    # 7. Equal-weight baseline  (no training)
    # -------------------------------------------------------
    print("\n--- BASELINE: equal-weight prototype fusion (no gate training) ---")
    base_X, base_y, base_u = build_train_embeddings_from_records(
        model, train_records, frozen_modalities, DEVICE
    )
    base_proto = compute_prototypes(base_X, base_y, num_classes)
    model.set_prototypes(base_proto)

    def _proto_pred(records_list, mod_names):
        preds, ys = [], []
        for r in records_list:
            seq = load_lip_seq(r["lm_dir"])
            if seq is None: continue
            vel  = compute_velocity(seq)
            feat = np.concatenate([seq, vel], axis=1)
            x    = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
            L    = torch.tensor([len(feat)], dtype=torch.long).to(DEVICE)
            with torch.no_grad():
                lip_emb = model.lip_enc(x, L).squeeze(0).cpu().numpy()
            frozen_e = np.stack([r["frozen_embs"][m] for m in mod_names])
            all_e    = np.concatenate([lip_emb[None], frozen_e], axis=0)[None]  # (1, K, 128)
            P        = base_proto
            all_e_n  = all_e / np.linalg.norm(all_e, axis=2, keepdims=True).clip(1e-8)
            sims     = np.einsum("bkd,kcd->bkc", all_e_n, P) * PROTO_TEMP
            sims    -= sims.max(axis=2, keepdims=True)
            probs    = np.exp(sims) / np.exp(sims).sum(axis=2, keepdims=True)
            fused    = probs.mean(axis=1)
            preds.append(fused.argmax(axis=1)[0])
            ys.append(r["label"])
        return np.array(preds), np.array(ys)

    va_p, va_y = _proto_pred(val_records,  frozen_modalities)
    te_p, te_y = _proto_pred(test_records, frozen_modalities)
    print(f"  Val:  {(va_p == va_y).mean():.3f}   Test: {(te_p == te_y).mean():.3f}")

    # -------------------------------------------------------
    # 8. Train
    # -------------------------------------------------------
    model = run_training(model, train_records, frozen_modalities,
                         train_loader, val_loader, num_classes, label_map)

    # -------------------------------------------------------
    # 9. Final evaluation
    # -------------------------------------------------------
    criterion = nn.NLLLoss()
    _, test_acc, test_preds, test_labels, test_gate_w = eval_one_epoch(
        model, test_loader, criterion
    )
    print(f"\nTest accuracy (users {TEST_USERS}): {test_acc:.3f}")
    idx_to_label = {v: k for k, v in label_map.items()}
    print(classification_report(
        test_labels, test_preds,
        target_names=[idx_to_label[i] for i in range(num_classes)],
        zero_division=0,
    ))

    _, val_acc, _, _, val_gate_w = eval_one_epoch(model, val_loader, criterion)
    print(f"Val accuracy: {val_acc:.3f}")

    all_mod_names = ["lip"] + frozen_modalities
    print("\nMean gate weight per modality:")
    all_w = np.concatenate([val_gate_w, test_gate_w])
    for i, m in enumerate(all_mod_names):
        print(f"  {m:12s}: {all_w[:, i].mean():.3f}")

    # -------------------------------------------------------
    # 10. Save fine-tuned lip embeddings and fused embeddings
    # -------------------------------------------------------
    print("\nSaving fine-tuned lip embeddings...")
    all_records  = train_records + val_records + test_records
    lip_embs, lip_labels, lip_uids, lip_gnames = [], [], [], []
    for r in all_records:
        seq = load_lip_seq(r["lm_dir"])
        if seq is None: continue
        vel  = compute_velocity(seq)
        feat = np.concatenate([seq, vel], axis=1)
        x    = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
        L    = torch.tensor([len(feat)], dtype=torch.long).to(DEVICE)
        with torch.no_grad():
            emb = model.lip_enc(x, L).squeeze(0).cpu().numpy()
        lip_embs.append(emb)
        lip_labels.append(r["label"])
        lip_uids.append(r["user_id"])
        lip_gnames.append(r["group_name"])

    np.savez_compressed(LIP_EMB_OUT_PATH,
                        embeddings=np.stack(lip_embs),
                        labels=np.array(lip_labels, dtype=np.int32),
                        user_ids=np.array(lip_uids),
                        group_names=np.array(lip_gnames))
    print(f"  Saved {len(lip_embs)} lip embeddings → {LIP_EMB_OUT_PATH}")

    print("\nDone. Re-run fusionGate.py with lip_embeddings_e2e.npz for the final fusion result.")


if __name__ == "__main__":
    main()