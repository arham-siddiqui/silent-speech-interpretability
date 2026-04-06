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
INCLUDE_PREFIXES = ["word"]   # change to ["vowel"] or ["sentences"] if needed
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


def get_samples(root, include_prefixes):
    """
    Returns a list of dicts, one per video sample.
    Each sample corresponds to:
    root/participant/category/videos/video_x/mouths/*.png
    """
    samples = []

    participant_dirs = sorted([
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
    ])

    for participant in participant_dirs:
        p_path = os.path.join(root, participant)

        category_dirs = sorted([
            d for d in os.listdir(p_path)
            if os.path.isdir(os.path.join(p_path, d))
        ])

        for category in category_dirs:
            if not wanted(category, include_prefixes):
                continue

            videos_dir = os.path.join(p_path, category, "videos")
            if not os.path.isdir(videos_dir):
                continue

            for video_name in sorted(os.listdir(videos_dir)):
                video_path = os.path.join(videos_dir, video_name)
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
                    "video_name": video_name,
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

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)

        header = [
            "participant",
            "label_name",
            "video_name",
            "mouths_dir",
            "num_frames",
        ] + [f"embed_{i}" for i in range(embedding_dim)]
        writer.writerow(header)

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
                print(f"Processed {idx + 1}/{len(samples)}")

    print(f"Saved embeddings to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()