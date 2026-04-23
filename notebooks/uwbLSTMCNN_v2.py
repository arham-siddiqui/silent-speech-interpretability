"""
uwbLSTMCNN_v2.py
================
Improved UWB radar encoder. The original uwbLSTMCNN.py reaches only ~22%
speaker-disjoint val accuracy. Three targeted improvements:

1. BETTER PREPROCESSING
   - Clip to ±3σ before normalisation (outlier RTM values hurt CNN weights)
   - Per-antenna per-bin z-score instead of global z-score
     (each range bin has different mean reflectivity; per-bin norm removes
      static background more completely than global)

2. TEMPORAL ATTENTION POOLING
   Same as the lip model: instead of the last BiLSTM hidden state, compute a
   weighted average over all LSTM outputs. Focuses on the frames where mouth
   motion produces the strongest radar backscatter.

3. DOMAIN ADVERSARIAL TRAINING  (DANN)
   Speaker-adversarial head via Gradient Reversal Layer — forces the CNN+LSTM
   to produce speaker-agnostic embeddings that generalise to unseen people.
   Combined loss: CE_loss + λ_dann * Speaker_adv_loss (annealed schedule).

ARCHITECTURE
------------
  Input (B, 2, 205, T_max)
  → 4-layer residual 2D CNN
      channels: 2→32→64→128→128, stride (2,1)×4 in time, ×2 in range
      range bins: 205→103→52→26→13
      time:       T→T/2→T/4→T/8→T/8  (same as before)
  → max-pool over range → (B, 128, T/8)
  → (B, T/8, 128) → BiLSTM (128 hidden/dir) → temporal attention → (B, 256)
  → Dropout → Linear(256→128) + LN → L2-norm  [EMBEDDING]
  → Linear(128 → num_classes)                  [classifier]
  → GRL → Linear(128→64)→ReLU→Linear(64→num_speakers)  [speaker adversarial]

OUTPUTS
-------
- uwb_cnn_lstm_model_v2.pt
- uwb_embeddings_v2.npz
- uwb_label_map_v2.json
"""

import os, json, re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report
from collections import defaultdict

# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src/data/RVTALL/Processed_cut_data/uwb_processed/")

MAX_USER_ID = 20
EXPECTED_RANGE_BINS = 205

HIDDEN_SIZE   = 128
NUM_LAYERS    = 1
EMBEDDING_DIM = 128
DROPOUT       = 0.3

BATCH_SIZE   = 16
LR           = 3e-4
EPOCHS       = 80
PATIENCE     = 25

LAMBDA_DANN = 0.3

VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]

MODEL_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "uwb_cnn_lstm_model_v2.pt")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "uwb_embeddings_v2.npz")
LABEL_MAP_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "uwb_label_map_v2.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# GRADIENT REVERSAL  (same as lip v2)
# ============================================================

class GradRevFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(alpha)
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha, = ctx.saved_tensors
        return -alpha * grad_output, None


def grad_reverse(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    a = torch.tensor(alpha, dtype=x.dtype, device=x.device)
    return GradRevFunction.apply(x, a)


def dann_alpha(epoch: int, total_epochs: int, max_alpha: float = LAMBDA_DANN) -> float:
    p = epoch / total_epochs
    return max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ============================================================
# DATA LOADING
# ============================================================

def normalise_group_name(folder_name: str) -> str:
    return folder_name.replace("_", "", 1)


def load_uwb_rtm(ant1_path: str, ant2_path: str):
    """
    Load both antenna RTMs, apply improved preprocessing, return (T, 2, 205).

    Improvements over v1:
    - Per-antenna per-range-bin z-score (removes static per-bin mean)
    - ±3σ clip before global scale (handles hot-pixel outliers)
    """
    rtm1 = np.load(ant1_path).astype(np.float32)
    rtm2 = np.load(ant2_path).astype(np.float32)

    if rtm1.shape[0] != EXPECTED_RANGE_BINS or rtm2.shape[0] != EXPECTED_RANGE_BINS:
        return None

    T = min(rtm1.shape[1], rtm2.shape[1])
    rtm1, rtm2 = rtm1[:, :T], rtm2[:, :T]

    rtm = np.stack([rtm1, rtm2], axis=0)   # (2, 205, T)

    # Per-antenna, per-range-bin detrend: remove temporal mean per bin
    rtm = rtm - rtm.mean(axis=2, keepdims=True)   # (2, 205, T)

    # Per-antenna, per-range-bin z-score: divide by temporal std per bin
    std_per_bin = rtm.std(axis=2, keepdims=True)
    rtm = rtm / np.where(std_per_bin > 1e-8, std_per_bin, 1.0)

    # Clip outliers to ±3σ (global, post per-bin norm)
    rtm = np.clip(rtm, -3.0, 3.0)

    # Final global scale so values are in [−1, 1] roughly
    global_std = rtm.std()
    if global_std > 1e-8:
        rtm = rtm / global_std

    rtm = rtm.transpose(2, 0, 1)   # (T, 2, 205)
    return None if T < 10 else rtm.astype(np.float32)


def build_sample_list(root: str) -> list:
    """Scan uwb_processed/ and return list of paired (ant1, ant2) sample dicts."""
    samples = []

    for user_id in sorted(os.listdir(root), key=lambda x: (0, int(x)) if x.isdigit() else (1, x)):
        user_path = os.path.join(root, user_id)
        if not os.path.isdir(user_path):
            continue
        try:
            uid_int = int(user_id)
        except ValueError:
            continue
        if uid_int > MAX_USER_ID:
            continue

        for group_folder in sorted(os.listdir(user_path)):
            group_path = os.path.join(user_path, group_folder)
            if not os.path.isdir(group_path):
                continue

            group_name = normalise_group_name(group_folder)
            if not any(group_name.startswith(p) for p in ["sentences", "vowel", "word"]):
                continue

            # Discover antenna files: {user}_{type}_{n}_{antenna}_{sample}.npy
            npy_files = [f for f in os.listdir(group_path) if f.endswith(".npy")]

            # Group by sample index
            pairs = defaultdict(dict)
            for fname in npy_files:
                parts = fname.replace(".npy", "").split("_")
                # Expect at least 5 parts: user type n antenna sample
                if len(parts) < 5:
                    continue
                antenna = parts[-2]   # "1" or "2"
                sample  = parts[-1]   # "1", "2", ...
                if antenna in ("1", "2"):
                    pairs[sample][antenna] = os.path.join(group_path, fname)

            for sample_id, ant_files in pairs.items():
                if "1" in ant_files and "2" in ant_files:
                    samples.append({
                        "user_id":    user_id,
                        "group_name": group_name,
                        "sample_id":  sample_id,
                        "ant1_path":  ant_files["1"],
                        "ant2_path":  ant_files["2"],
                        "label_str":  group_name,
                    })

    print(f"Found {len(samples)} paired UWB samples.")
    return samples


# ============================================================
# DATASET
# ============================================================

class UWBDataset(Dataset):
    def __init__(self, samples, label_map, speaker_map, augment=False):
        self.augment = augment
        self.items   = []

        skipped = 0
        for s in samples:
            rtm = load_uwb_rtm(s["ant1_path"], s["ant2_path"])
            if rtm is None:
                skipped += 1
                continue
            lbl = label_map[s["label_str"]]
            spk = speaker_map.get(s["user_id"], 0)
            self.items.append((rtm, lbl, spk, s["sample_id"]))

        print(f"  Loaded {len(self.items)} samples ({skipped} skipped).")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        rtm, label, speaker, sid = self.items[idx]

        if self.augment:
            # Temporal crop: keep 80–100% of frames
            T = rtm.shape[0]
            if T > 20:
                keep  = int(np.random.uniform(0.80, 1.0) * T)
                start = np.random.randint(0, T - keep + 1)
                rtm   = rtm[start:start + keep]

            # Small additive noise
            rtm = rtm + np.random.normal(0, 0.02, rtm.shape).astype(np.float32)

        return torch.from_numpy(rtm), label, speaker, sid


def collate_fn(batch):
    rtms, labels, speakers, sids = zip(*batch)
    lengths  = torch.tensor([r.shape[0] for r in rtms], dtype=torch.long)
    padded   = pad_sequence(rtms, batch_first=True)   # (B, T_max, 2, 205)
    labels   = torch.tensor(labels,   dtype=torch.long)
    speakers = torch.tensor(speakers, dtype=torch.long)
    return padded, lengths, labels, speakers, list(sids)


# ============================================================
# CNN BLOCK WITH RESIDUAL
# ============================================================

class ResBlock2D(nn.Module):
    """Conv-BN-GELU-Conv-BN with stride-1 residual."""
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )

    def forward(self, x):
        return F.gelu(x + self.net(x))


# ============================================================
# MODEL
# ============================================================

# CNN stride schedule for range dimension per layer; time dim always gets stride=2
RANGE_STRIDES = [2, 2, 2, 1]
# The time dimension is always strided by 2 in every conv layer (stride=(range_s, 2))
TIME_STRIDE_PER_LAYER = 2
NUM_CONV_LAYERS = 4

class UWBEncoderV2(nn.Module):
    """
    Improved UWB encoder:
      - ResBlock2D instead of plain conv
      - Temporal attention pooling
      - Domain adversarial head
    """
    def __init__(self, num_classes, num_speakers,
                 hidden_size=128, embedding_dim=128, dropout=0.3):
        super().__init__()
        self.num_classes  = num_classes
        self.num_speakers = num_speakers
        self.NUM_CONV_LAYERS = NUM_CONV_LAYERS

        # 4-layer CNN: 2→32→64→128→128
        # stride=(range_stride, time_stride); time_stride is always 2
        def conv_block(in_ch, out_ch, range_stride):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=(3, 3), stride=(range_stride, 2), padding=(1, 1), bias=False),
                nn.BatchNorm2d(out_ch),
                nn.GELU(),
                ResBlock2D(out_ch),
            )

        self.cnn = nn.Sequential(
            conv_block(2,   32,  RANGE_STRIDES[0]),
            conv_block(32,  64,  RANGE_STRIDES[1]),
            conv_block(64,  128, RANGE_STRIDES[2]),
            conv_block(128, 128, RANGE_STRIDES[3]),
        )

        # BiLSTM over time
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_size,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.0,
        )

        self.attn_proj = nn.Linear(hidden_size * 2, 1, bias=False)

        self.dropout    = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        self.classifier  = nn.Linear(embedding_dim, num_classes)
        self.speaker_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_speakers),
        )

    def _time_out_lengths(self, lengths):
        # Every conv layer applies time_stride=2 (the width stride in Conv2d)
        L = lengths.float()
        for _ in range(self.NUM_CONV_LAYERS):
            L = torch.floor((L + 2 - 3) / 2 + 1)
        return L.long().clamp(min=1)

    def _attend(self, lstm_out, lengths):
        """Temporal attention pooling over LSTM outputs."""
        scores = self.attn_proj(lstm_out).squeeze(-1)   # (B, T)
        max_T  = lstm_out.size(1)
        mask   = torch.arange(max_T, device=lstm_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        w = F.softmax(scores, dim=1).unsqueeze(-1)      # (B, T, 1)
        return (lstm_out * w).sum(dim=1)                # (B, hidden*2)

    def forward(self, x, lengths, dann_alpha: float = 0.0):
        """
        x:       (B, T_max, 2, 205)
        lengths: (B,)
        """
        B, T_max, C, R = x.shape
        # CNN expects (B, C, R, T) — treat range as height, time as width
        x = x.permute(0, 2, 3, 1)   # (B, 2, 205, T_max)
        x = self.cnn(x)              # (B, 128, R', T')
        x = x.amax(dim=2)           # max-pool over range → (B, 128, T')
        x = x.permute(0, 2, 1)      # (B, T', 128)

        cnn_lengths = self._time_out_lengths(lengths)
        packed = pack_padded_sequence(x, cnn_lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # (B, T', 256)

        pooled  = self._attend(lstm_out, cnn_lengths)   # (B, 256)
        pooled  = self.dropout(pooled)
        raw_emb = self.embed_proj(pooled)               # (B, 128)

        class_logits   = self.classifier(raw_emb)
        rev_emb        = grad_reverse(raw_emb, alpha=dann_alpha)
        speaker_logits = self.speaker_head(rev_emb)
        embedding      = F.normalize(raw_emb, p=2, dim=1)

        return class_logits, speaker_logits, embedding

    def encode(self, x, lengths):
        with torch.no_grad():
            _, _, emb = self.forward(x, lengths, dann_alpha=0.0)
        return emb


# ============================================================
# TRAINING
# ============================================================

def train_epoch(model, loader, optimizer, criterion, epoch, total_epochs):
    model.train()
    alpha = dann_alpha(epoch, total_epochs)
    total_loss, correct, total = 0.0, 0, 0

    for padded, lengths, labels, speakers, _ in loader:
        padded, lengths, labels, speakers = (
            padded.to(DEVICE), lengths.to(DEVICE),
            labels.to(DEVICE), speakers.to(DEVICE)
        )
        optimizer.zero_grad()
        class_logits, speaker_logits, _ = model(padded, lengths, dann_alpha=alpha)
        ce_loss  = criterion(class_logits, labels)
        spk_loss = F.cross_entropy(speaker_logits, speakers)
        loss = ce_loss + alpha * spk_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += ce_loss.item() * len(labels)
        correct    += (class_logits.argmax(1) == labels).sum().item()
        total      += len(labels)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for padded, lengths, labels, _, _ in loader:
        padded, lengths, labels = (
            padded.to(DEVICE), lengths.to(DEVICE), labels.to(DEVICE)
        )
        class_logits, _, _ = model(padded, lengths, dann_alpha=0.0)
        loss    = criterion(class_logits, labels)
        total_loss += loss.item() * len(labels)
        preds   = class_logits.argmax(1)
        correct += (preds == labels).sum().item()
        total   += len(labels)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    return total_loss / total, correct / total, all_preds, all_labels


def train(model, train_loader, val_loader, label_map):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=10, min_lr=1e-5
    )

    best_val_acc, patience_ctr = 0.0, 0
    idx_to_label = {v: k for k, v in label_map.items()}

    print("\n" + "=" * 65)
    print("TRAINING UWB v2  (CE + domain adversarial, attention pooling)")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, EPOCHS)
        val_loss, val_acc, vp, vl = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)

        alpha  = dann_alpha(epoch, EPOCHS)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train {tr_acc:.3f} | Val {val_acc:.3f} | "
              f"dann_α={alpha:.3f} lr={lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ Best val: {best_val_acc:.3f}")
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    _, final_val_acc, vp, vl = eval_epoch(model, val_loader, criterion)
    print(f"\nFinal val acc: {final_val_acc:.3f}")
    target_names = [idx_to_label[i] for i in sorted(idx_to_label)]
    print(classification_report(vl, vp, target_names=target_names, zero_division=0))
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
        x = torch.from_numpy(rtm).unsqueeze(0).to(DEVICE)
        L = torch.tensor([rtm.shape[0]], dtype=torch.long).to(DEVICE)
        emb = model.encode(x, L).squeeze(0).cpu().numpy()
        embeddings.append(emb)
        labels.append(label_map[s["label_str"]])
        user_ids.append(s["user_id"])
        group_names.append(s["group_name"])
        sample_names.append(s["sample_id"])

    print(f"  Extracted {len(embeddings)} embeddings ({skipped} skipped).")
    return (
        np.stack(embeddings).astype(np.float32),
        np.array(labels,      dtype=np.int32),
        np.array(user_ids),
        np.array(group_names),
        np.array(sample_names),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    all_samples = build_sample_list(ROOT)
    if not all_samples:
        print("No samples found. Check ROOT path.")
        return

    unique_labels = sorted(set(s["label_str"] for s in all_samples))
    label_map  = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(label_map)
    print(f"\n{num_classes} classes")
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    train_samples = [s for s in all_samples if s["user_id"] not in VAL_USERS + TEST_USERS]
    val_samples   = [s for s in all_samples if s["user_id"] in VAL_USERS]
    test_samples  = [s for s in all_samples if s["user_id"] in TEST_USERS]

    train_users  = sorted(set(s["user_id"] for s in train_samples),
                          key=lambda u: int(u) if u.isdigit() else u)
    speaker_map  = {u: i for i, u in enumerate(train_users)}
    num_speakers = len(train_users)
    print(f"Training speakers ({num_speakers}): {train_users}")
    print(f"Split: {len(train_samples)} train | {len(val_samples)} val | {len(test_samples)} test")

    print("\nLoading training data...")
    train_ds = UWBDataset(train_samples, label_map, speaker_map, augment=True)
    print("Loading val data...")
    val_ds   = UWBDataset(val_samples,   label_map, speaker_map, augment=False)
    print("Loading test data...")
    test_ds  = UWBDataset(test_samples,  label_map, speaker_map, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = UWBEncoderV2(
        num_classes=num_classes,
        num_speakers=num_speakers,
        hidden_size=HIDDEN_SIZE,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")

    model = train(model, train_loader, val_loader, label_map)

    criterion = nn.CrossEntropyLoss()
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
    print(f"\nTest accuracy (users {TEST_USERS}): {test_acc:.3f}")
    idx_to_label = {v: k for k, v in label_map.items()}
    print(classification_report(
        test_labels, test_preds,
        target_names=[idx_to_label[i] for i in sorted(idx_to_label)],
        zero_division=0,
    ))

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
    print(f"\nEmbeddings saved to {EMBEDDINGS_PATH}  shape={embs.shape}")


if __name__ == "__main__":
    main()