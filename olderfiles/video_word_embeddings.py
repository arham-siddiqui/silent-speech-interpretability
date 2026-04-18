import os
import csv
from PIL import Image

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

import ssl
import certifi

ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


ROOT = "src/data/RVTALL/Processed_cut_data/kinect_processed"
OUTPUT_CSV = "mouth_frame_embeddings.csv"
INCLUDE_PREFIXES = ["sentences", "vowel"]   # words already extracted
IMG_SIZE = 96
BATCH_SIZE = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_transform(size=96):
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])


def wanted(category_name, include_prefixes):
    return any(category_name.startswith(p) for p in include_prefixes)


def numeric_participant_key(name):
    """Sort participants numerically: 1,2,3...20 not 1,10,11...2."""
    import re
    nums = re.findall(r"\d+", name)
    return int(nums[0]) if nums else 0


def category_sort_key(name):
    """Sort categories: sentences1-10 then vowel1-5 (matching lip NPZ order)."""
    import re
    for i, prefix in enumerate(["sentences", "vowel", "word"]):
        if name.startswith(prefix):
            nums = re.findall(r"\d+", name)
            return (i, int(nums[0]) if nums else 0)
    return (99, 0)


def get_samples(root, include_prefixes):
    """
    Returns a list of dicts, one per video sample.
    Participants sorted numerically (1,2,...,20).
    Categories sorted as sentences1-10, vowel1-5 to match lip NPZ order.
    Handles participant 1 → video_* naming, others → video_proc_* naming.
    """
    import glob as _glob
    samples = []

    participant_dirs = sorted(
        [d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))],
        key=numeric_participant_key
    )

    for participant in participant_dirs:
        p_path = os.path.join(root, participant)

        category_dirs = sorted(
            [d for d in os.listdir(p_path) if os.path.isdir(os.path.join(p_path, d))],
            key=category_sort_key
        )

        for category in category_dirs:
            if not wanted(category, include_prefixes):
                continue

            videos_dir = os.path.join(p_path, category, "videos")
            if not os.path.isdir(videos_dir):
                continue

            # Participant 1 uses video_N style; 2-20 use video_proc_N style
            if participant == "1":
                video_dirs = sorted(
                    _glob.glob(os.path.join(videos_dir, "video_[0-9]*")),
                    key=lambda p: numeric_participant_key(os.path.basename(p))
                )
            else:
                video_dirs = sorted(
                    _glob.glob(os.path.join(videos_dir, "video_proc_*")),
                    key=lambda p: numeric_participant_key(os.path.basename(p))
                )

            for video_path in video_dirs:
                if not os.path.isdir(video_path):
                    continue

                mouths_dir = os.path.join(video_path, "mouths")
                if not os.path.isdir(mouths_dir):
                    continue

                pngs = sorted([
                    os.path.join(mouths_dir, f)
                    for f in os.listdir(mouths_dir)
                    if f.lower().endswith(".png")
                ])

                if len(pngs) == 0:
                    continue

                samples.append({
                    "participant": participant,
                    "label_name": category,
                    "video_name": os.path.basename(video_path),
                    "mouths_dir": mouths_dir,
                    "frame_paths": pngs,
                })

    return samples


def build_encoder():
    """
    Pretrained ResNet18 with final classifier removed.
    Output embedding size = 512
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    encoder = nn.Sequential(*list(model.children())[:-1])  # remove final FC
    encoder.eval()
    encoder.to(DEVICE)
    return encoder, weights


@torch.no_grad()
def extract_video_embedding(frame_paths, encoder, transform, batch_size=32):
    """
    Returns one embedding for one video by averaging frame embeddings.
    Output shape: [512]
    """
    frame_tensors = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        frame_tensors.append(transform(img))

    embeddings = []

    for i in range(0, len(frame_tensors), batch_size):
        batch = torch.stack(frame_tensors[i:i+batch_size]).to(DEVICE)   # [B, 3, H, W]
        feats = encoder(batch)                                          # [B, 512, 1, 1]
        feats = feats.squeeze(-1).squeeze(-1)                           # [B, 512]
        embeddings.append(feats.cpu())

    embeddings = torch.cat(embeddings, dim=0)                           # [num_frames, 512]
    video_embedding = embeddings.mean(dim=0)                            # [512]
    return video_embedding


def main():
    print("Scanning samples...")
    samples = get_samples(ROOT, INCLUDE_PREFIXES)
    print(f"Found {len(samples)} samples")

    encoder, weights = build_encoder()
    transform = get_transform(IMG_SIZE)

    if len(samples) == 0:
        print("No samples found. Check ROOT and folder structure.")
        return

    embedding_dim = 512

    # APPEND — words already written; do not re-write header
    with open(OUTPUT_CSV, "a", newline="") as f:
        writer = csv.writer(f)

        for idx, sample in enumerate(samples):
            emb = extract_video_embedding(
                sample["frame_paths"],
                encoder,
                transform,
                batch_size=BATCH_SIZE
            )

            row = [
                sample["participant"],
                sample["label_name"],
                sample["video_name"],
                sample["mouths_dir"],
                len(sample["frame_paths"]),
            ] + emb.tolist()

            writer.writerow(row)

            if (idx + 1) % 50 == 0 or idx == len(samples) - 1:
                import time as _time
                print(f"[{_time.strftime('%H:%M:%S')}] Processed {idx + 1}/{len(samples)} "
                      f"({sample['participant']}/{sample['label_name']}/{sample['video_name']})",
                      flush=True)

    print(f"\nAppended {len(samples)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()