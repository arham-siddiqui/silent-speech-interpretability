"""
liplandmarkLSTM_v2.py
=====================
Improved lip landmark encoder with three changes to push past the
~25% speaker-disjoint ceiling of the original model:

1. TEMPORAL ATTENTION POOLING
   Instead of taking only the last BiLSTM hidden state, compute a
   learned weighted average over ALL timestep outputs. This lets the
   model focus on the most discriminative frames (peak articulation)
   rather than being dominated by the final frame.

2. SUPERVISED CONTRASTIVE LOSS  (SupCon, Khosla et al. 2020)
   In addition to cross-entropy, a contrastive loss pulls embeddings
   of the same utterance type together — across DIFFERENT speakers.
   This directly optimises for speaker-invariant clustering, which is
   exactly what the fusion layer needs.
   Combined loss: CE_loss + λ_sc * SupCon_loss

3. DOMAIN ADVERSARIAL TRAINING  (DANN, Ganin et al. 2016)
   A speaker-ID classifier is attached via a Gradient Reversal Layer.
   The encoder is rewarded for confusing the speaker classifier →
   embeddings become speaker-agnostic.
   Loss: CE_loss + λ_sc * SupCon_loss - λ_dann * Speaker_CE_loss
   λ_dann anneals from 0→1 over training (standard DANN schedule).

WHY THESE THREE TOGETHER
-------------------------
- SupCon: explicit cross-speaker class clustering in embedding space
- DANN: implicit speaker-feature removal from embedding
- Attention: focuses on phonetically informative frames, not noise

OUTPUTS
-------
- lip_lstm_model_v2.pt
- lip_embeddings_v2.npz  ← use this in fusionGate.py / fusionE2E.py
- lip_label_map_v2.json
"""

import os, re, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report

# ============================================================
# CONFIG
# ============================================================

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "src/data/RVTALL/Processed_cut_data/kinect_processed/")

LIP_START, LIP_END = 48, 68   # dlib 68-point → 20 lip points

HIDDEN_SIZE   = 256
NUM_LAYERS    = 2
EMBEDDING_DIM = 128
DROPOUT       = 0.3

BATCH_SIZE   = 32
LR           = 3e-4
EPOCHS       = 80
PATIENCE     = 25

LAMBDA_SC   = 0.5    # SupCon weight
LAMBDA_DANN = 0.3    # Speaker adversarial weight (full scale — annealed during training)
SC_TEMP     = 0.07   # SupCon temperature

VAL_USERS  = ["17", "18"]
TEST_USERS = ["19", "20"]

MODEL_PATH      = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lip_lstm_model_v2.pt")
EMBEDDINGS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lip_embeddings_v2.npz")
LABEL_MAP_PATH  = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "lip_label_map_v2.json")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")


# ============================================================
# GRADIENT REVERSAL LAYER  (DANN)
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


# ============================================================
# SUPERVISED CONTRASTIVE LOSS
# ============================================================

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. 2020).
    For each anchor, positives = same label, negatives = different label.
    Gradients pull same-class embeddings together across all speakers.
    """
    def __init__(self, temperature: float = SC_TEMP):
        super().__init__()
        self.temp = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # features: (B, D) L2-normalised
        # labels:   (B,)
        B, D = features.shape
        device = features.device

        # Pairwise cosine similarity (features already L2-normed)
        sims = torch.mm(features, features.T) / self.temp   # (B, B)

        # Numerical stability: subtract row max (does not change softmax)
        sims = sims - sims.max(dim=1, keepdim=True).values.detach()

        eye = torch.eye(B, dtype=torch.bool, device=device)
        # Positive mask: same label, NOT same sample
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)) & ~eye  # (B, B)

        if pos_mask.sum() == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # exp of all non-self pairs
        exp_sims = torch.exp(sims) * (~eye).float()   # (B, B), diagonal = 0
        # log-softmax denominator = log sum of all non-self exp(sim)
        log_denom = torch.log(exp_sims.sum(dim=1, keepdim=True) + 1e-8)   # (B, 1)
        log_prob  = sims - log_denom   # (B, B)

        # Mean log-prob over positive pairs per anchor
        n_pos = pos_mask.float().sum(dim=1).clamp(min=1)
        per_sample = (log_prob * pos_mask.float()).sum(dim=1) / n_pos   # (B,)
        return -per_sample.mean()


# ============================================================
# DATA LOADING
# ============================================================

def list_sorted_npy_files(directory):
    files = glob.glob(os.path.join(directory, "*.npy"))
    def num_key(p):
        ns = re.findall(r"\d+", os.path.basename(p))
        return [int(n) for n in ns] if ns else [os.path.basename(p)]
    return sorted(files, key=num_key)


def normalize_landmarks(lm):
    lm = np.asarray(lm, dtype=np.float32)
    centroid = lm.mean(axis=0)
    lm -= centroid
    scale = np.max(np.linalg.norm(lm, axis=1)) + 1e-8
    return lm / scale


def load_lip_sequence(landmarkers_dir):
    files = list_sorted_npy_files(landmarkers_dir)
    frames = []
    for f in files:
        arr = np.load(f)
        if arr.ndim != 2 or arr.shape[0] < 68 or arr.shape[1] < 2:
            continue
        lip = arr[LIP_START:LIP_END, :2]
        lip = normalize_landmarks(lip)
        frames.append(lip.flatten())
    if len(frames) < 5:
        return None
    return np.asarray(frames, dtype=np.float32)  # (T, 40)


def compute_velocity(seq):
    return np.gradient(seq, axis=0).astype(np.float32)


def build_sample_list(root):
    samples = []
    user_dirs = sorted(
        [d for d in glob.glob(os.path.join(root, "*")) if os.path.isdir(d)],
        key=lambda p: int(re.findall(r"\d+", os.path.basename(p))[0])
        if re.findall(r"\d+", os.path.basename(p)) else os.path.basename(p)
    )
    if not user_dirs:
        raise RuntimeError(f"No user directories found under: {os.path.abspath(root)}")

    for user_dir in user_dirs:
        user_id = os.path.basename(user_dir)
        for group_prefix in ["sentences", "vowel", "word"]:
            for group_dir in sorted(glob.glob(os.path.join(user_dir, f"{group_prefix}*"))):
                group_name = os.path.basename(group_dir)
                videos_dir = os.path.join(group_dir, "videos")
                if not os.path.isdir(videos_dir):
                    continue
                vid_glob = "video_[0-9]*" if user_id == "1" else "video_proc_*"
                for video_dir in sorted(glob.glob(os.path.join(videos_dir, vid_glob))):
                    lm_dir = os.path.join(video_dir, "landmarkers_cv")
                    if not os.path.isdir(lm_dir):
                        continue
                    samples.append({
                        "user_id":        user_id,
                        "group_name":     group_name,
                        "video_name":     os.path.basename(video_dir),
                        "landmarkers_dir": lm_dir,
                        "label_str":      group_name,
                    })
    print(f"Found {len(samples)} candidate samples across {len(user_dirs)} users.")
    return samples


# ============================================================
# DATASET
# ============================================================

class LipDataset(Dataset):
    def __init__(self, samples, label_map, speaker_map, augment=False):
        self.label_map   = label_map
        self.speaker_map = speaker_map
        self.augment     = augment
        self.items       = []   # (seq, label_int, speaker_int)

        skipped = 0
        for s in samples:
            seq = load_lip_sequence(s["landmarkers_dir"])
            if seq is None:
                skipped += 1
                continue
            vel  = compute_velocity(seq)
            feat = np.concatenate([seq, vel], axis=1)  # (T, 80)
            lbl  = label_map[s["label_str"]]
            spk  = speaker_map.get(s["user_id"], 0)
            self.items.append((feat, lbl, spk, s["video_name"]))

        print(f"  Loaded {len(self.items)} samples ({skipped} skipped).")

    def __len__(self): return len(self.items)

    def __getitem__(self, idx):
        seq, label, speaker, vname = self.items[idx]

        if self.augment:
            T = len(seq)
            if T > 10:
                keep  = int(np.random.uniform(0.75, 1.0) * T)
                start = np.random.randint(0, T - keep + 1)
                seq   = seq[start:start + keep]

            # Speed perturbation: sub-sample or interpolate ±15%
            if np.random.rand() < 0.5:
                speed = np.random.uniform(0.85, 1.15)
                new_T = max(5, int(len(seq) / speed))
                idx_f = np.linspace(0, len(seq) - 1, new_T)
                seq   = np.stack([np.interp(idx_f, np.arange(len(seq)), seq[:, d])
                                   for d in range(seq.shape[1])], axis=1).astype(np.float32)

            # Gaussian noise
            seq = seq + np.random.normal(0, 0.005, seq.shape).astype(np.float32)

        return torch.from_numpy(seq), label, speaker, vname


def collate_fn(batch):
    seqs, labels, speakers, vnames = zip(*batch)
    lengths  = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    padded   = pad_sequence(seqs, batch_first=True)
    labels   = torch.tensor(labels,   dtype=torch.long)
    speakers = torch.tensor(speakers, dtype=torch.long)
    return padded, lengths, labels, speakers, list(vnames)


# ============================================================
# MODEL
# ============================================================

class LipLSTMV2(nn.Module):
    """
    BiLSTM with temporal attention pooling + domain adversarial head.

    Architecture:
      (T, 80) → LayerNorm → BiLSTM(256×2) → temporal attention → (512,)
              → Dropout → Linear→LN → (128,) [raw embedding]
              → L2-norm                       [for SupCon + fusion]
              → Linear(128 → num_classes)     [utterance classifier]
              → grad_reverse → Linear(128 → num_speakers)  [speaker adv.]
    """
    def __init__(self, input_size, num_classes, num_speakers,
                 hidden_size=256, num_layers=2, embedding_dim=128, dropout=0.3):
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

        # Temporal attention: scalar score per timestep
        self.attn_proj = nn.Linear(hidden_size * 2, 1, bias=False)

        self.dropout    = nn.Dropout(dropout)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size * 2, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

        # Utterance classifier
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Speaker adversarial head
        self.speaker_head = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_speakers),
        )

    def _attend(self, lstm_out: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        lstm_out: (B, T, 512)
        lengths:  (B,)
        Returns:  (B, 512) attention-pooled
        """
        scores = self.attn_proj(lstm_out).squeeze(-1)   # (B, T)
        # Mask padded positions
        max_T  = lstm_out.size(1)
        mask   = torch.arange(max_T, device=lstm_out.device).unsqueeze(0) >= lengths.unsqueeze(1)
        scores = scores.masked_fill(mask, float("-inf"))
        weights = F.softmax(scores, dim=1).unsqueeze(-1)   # (B, T, 1)
        return (lstm_out * weights).sum(dim=1)             # (B, 512)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor, dann_alpha: float = 0.0):
        """
        x:          (B, T_max, 80)
        lengths:    (B,)
        dann_alpha: gradient reversal scale (0 at start, anneals to LAMBDA_DANN)
        Returns:
            class_logits:   (B, C)
            speaker_logits: (B, S)  (via GRL)
            embedding:      (B, 128) L2-normed
        """
        x = self.input_norm(x)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # (B, T, 512)

        pooled    = self._attend(lstm_out, lengths)    # (B, 512)
        pooled    = self.dropout(pooled)
        raw_emb   = self.embed_proj(pooled)            # (B, 128)

        class_logits   = self.classifier(raw_emb)

        # Domain adversarial: gradient reversal on embedding
        rev_emb        = grad_reverse(raw_emb, alpha=dann_alpha)
        speaker_logits = self.speaker_head(rev_emb)

        embedding      = F.normalize(raw_emb, p=2, dim=1)
        return class_logits, speaker_logits, embedding

    def encode(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            _, _, emb = self.forward(x, lengths, dann_alpha=0.0)
        return emb


# ============================================================
# DANN ALPHA SCHEDULE
# ============================================================

def dann_alpha(epoch: int, total_epochs: int, max_alpha: float = LAMBDA_DANN) -> float:
    """Smooth annealing from 0 → max_alpha using the standard DANN schedule."""
    p = epoch / total_epochs
    return max_alpha * (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)


# ============================================================
# TRAINING
# ============================================================

supcon_loss_fn = SupConLoss(temperature=SC_TEMP)


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

        class_logits, speaker_logits, embedding = model(padded, lengths, dann_alpha=alpha)

        ce_loss   = criterion(class_logits, labels)
        sc_loss   = supcon_loss_fn(embedding, labels)
        spk_loss  = F.cross_entropy(speaker_logits, speakers)

        loss = ce_loss + LAMBDA_SC * sc_loss + alpha * spk_loss
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += ce_loss.item() * len(labels)   # track CE for readability
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
    print("TRAINING  (CE + SupCon + domain adversarial)")
    print("=" * 65)

    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, optimizer, criterion, epoch, EPOCHS)
        val_loss, val_acc, val_preds, val_labels_list = eval_epoch(model, val_loader, criterion)
        scheduler.step(val_acc)

        alpha = dann_alpha(epoch, EPOCHS)
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch:3d}/{EPOCHS} | "
              f"Train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
              f"Val loss {val_loss:.4f} acc {val_acc:.3f} | "
              f"dann_α={alpha:.3f} lr={lr_now:.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_ctr = 0
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  ✓ New best val acc: {best_val_acc:.3f} — saved.")
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
    embeddings, labels, user_ids, group_names, video_names = [], [], [], [], []
    skipped = 0

    print("\nExtracting embeddings for all samples...")
    for s in all_samples:
        seq = load_lip_sequence(s["landmarkers_dir"])
        if seq is None:
            skipped += 1
            continue
        vel  = compute_velocity(seq)
        feat = np.concatenate([seq, vel], axis=1)
        x    = torch.from_numpy(feat).unsqueeze(0).to(DEVICE)
        L    = torch.tensor([len(feat)], dtype=torch.long).to(DEVICE)
        emb  = model.encode(x, L).squeeze(0).cpu().numpy()
        embeddings.append(emb)
        labels.append(label_map[s["label_str"]])
        user_ids.append(s["user_id"])
        group_names.append(s["group_name"])
        video_names.append(s["video_name"])

    print(f"  Extracted {len(embeddings)} embeddings ({skipped} skipped).")
    return (
        np.stack(embeddings).astype(np.float32),
        np.array(labels,      dtype=np.int32),
        np.array(user_ids),
        np.array(group_names),
        np.array(video_names),
    )


# ============================================================
# MAIN
# ============================================================

def main():
    all_samples = build_sample_list(ROOT)
    if not all_samples:
        print("No samples found. Check ROOT path.")
        return

    # Label map
    unique_labels = sorted(set(s["label_str"] for s in all_samples))
    label_map = {lbl: i for i, lbl in enumerate(unique_labels)}
    num_classes = len(label_map)
    print(f"\n{num_classes} classes")
    with open(LABEL_MAP_PATH, "w") as f:
        json.dump(label_map, f, indent=2)

    # Speaker map  (only training speakers)
    train_samples = [s for s in all_samples if s["user_id"] not in VAL_USERS + TEST_USERS]
    val_samples   = [s for s in all_samples if s["user_id"] in VAL_USERS]
    test_samples  = [s for s in all_samples if s["user_id"] in TEST_USERS]

    train_users   = sorted(set(s["user_id"] for s in train_samples),
                           key=lambda u: int(u) if u.isdigit() else u)
    speaker_map   = {u: i for i, u in enumerate(train_users)}
    num_speakers  = len(train_users)
    print(f"Training speakers ({num_speakers}): {train_users}")
    print(f"Split: {len(train_samples)} train | {len(val_samples)} val | {len(test_samples)} test")

    # Datasets
    print("\nLoading training data...")
    train_ds = LipDataset(train_samples, label_map, speaker_map, augment=True)
    print("Loading val data...")
    val_ds   = LipDataset(val_samples,   label_map, speaker_map, augment=False)
    print("Loading test data...")
    test_ds  = LipDataset(test_samples,  label_map, speaker_map, augment=False)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    # Model
    model = LipLSTMV2(
        input_size=80,
        num_classes=num_classes,
        num_speakers=num_speakers,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        embedding_dim=EMBEDDING_DIM,
        dropout=DROPOUT,
    ).to(DEVICE)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters: {total_params:,}")
    print(f"Improvements: attention pooling + SupCon (λ={LAMBDA_SC}) + DANN (λ={LAMBDA_DANN})")

    # Train
    model = train(model, train_loader, val_loader, label_map)

    # Test evaluation
    criterion = nn.CrossEntropyLoss()
    _, test_acc, test_preds, test_labels = eval_epoch(model, test_loader, criterion)
    print(f"\nTest accuracy (users {TEST_USERS}): {test_acc:.3f}")
    idx_to_label = {v: k for k, v in label_map.items()}
    print(classification_report(
        test_labels, test_preds,
        target_names=[idx_to_label[i] for i in sorted(idx_to_label)],
        zero_division=0,
    ))

    # Extract and save all embeddings
    embs, labels_arr, user_ids, group_names, video_names = \
        extract_all_embeddings(model, all_samples, label_map)

    np.savez_compressed(
        EMBEDDINGS_PATH,
        embeddings=embs,
        labels=labels_arr,
        user_ids=user_ids,
        group_names=group_names,
        video_names=video_names,
    )
    print(f"\nEmbeddings saved to {EMBEDDINGS_PATH}  shape={embs.shape}")


if __name__ == "__main__":
    main()