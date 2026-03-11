import os
import argparse
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision.models as models
import torchvision.transforms as T



DEFAULT_CFG = {
    "T":            16,              
    "H":            96,              
    "W":            96,              
    "C":            3,                
    "embed_dim":    256,             
    "tcn_channels": [256, 256, 256],  
    "tcn_kernel":   3,
    "dropout":      0.2,
    "lr":           1e-4,
    "batch_size":   16,
    "epochs":       30,
    "num_workers":  4,
}

WORD_PREFIXES     = ["word"]
VOWEL_PREFIXES    = ["vowel"]
SENTENCE_PREFIXES = ["sentences"]


def get_frame_transform(H=96, W=96, augment=True):
    """Transform for a mouth PNG"""
    base = [
        T.Resize((H, W)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),   
    ]
    if augment:
        aug = [
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.3, contrast=0.3),
        ]
        return T.Compose(aug + base)
    return T.Compose(base)


def _get_mouth_frames(category_path):
    """
    Given  .../participant/word1/
    Returns the sorted list of PNG paths inside  .../word1/videos/mouths/
    Returns [] if the mouths folder doesn't exist or is empty.
    """
    mouths_dir = os.path.join(category_path, "videos", "mouths")
    if not os.path.isdir(mouths_dir):
        return []
    return sorted([
        os.path.join(mouths_dir, f)
        for f in os.listdir(mouths_dir)
        if f.lower().endswith(".png")
    ])

""" For KinectMouthDataset file"""

class KinectMouthDataset(Dataset):

    def __init__(self, root, T=16, H=96, W=96,
                 include_prefixes=None,
                 transform=None,
                 label_map=None):
        self.T         = T
        self.H         = H
        self.W         = W
        self.transform = transform or get_frame_transform(H, W, augment=True)
        self.include_prefixes = include_prefixes or WORD_PREFIXES
        self.label_map = label_map
        self.samples   = []   
        self._scan(root)

    def _wanted(self, category_name):
        """Return True if this category matches one of our include prefixes."""
        return any(category_name.startswith(p) for p in self.include_prefixes)

    def _scan(self, root):
        raw = []
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
                if not self._wanted(category):
                    continue
                pngs = _get_mouth_frames(os.path.join(p_path, category))
                if pngs:
                    raw.append((pngs, category))

        if self.label_map is None:
            all_categories = sorted(set(cat for _, cat in raw))
            self.label_map = {cat: i for i, cat in enumerate(all_categories)}

        for pngs, category in raw:
            label = self.label_map.get(category, -1)
            if label >= 0:
                self.samples.append((pngs, label))

        print(f"[Dataset] {len(self.samples)} samples | "
              f"{len(self.label_map)} classes: {list(self.label_map.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        png_paths, label = self.samples[idx]

        frames = []
        for path in png_paths:
            img = Image.open(path).convert("RGB")
            frames.append(self.transform(img))  

        clip = torch.stack(frames, dim=1)       
        clip = self._pad_or_trim(clip)           

        return clip.float(), torch.tensor(label, dtype=torch.long)