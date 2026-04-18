# dataset_png_sequence_direct.py
import os
import torch
from PIL import Image
from torch.utils.data import Dataset

class MouthSequenceDataset(Dataset):
    def __init__(self, root_dir, num_frames=16, image_size=96, grayscale=True):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.image_size = image_size
        self.grayscale = grayscale

        self.samples = []
        self._build_samples()

        if len(self.samples) == 0:
            raise ValueError(f"No valid mouths folders found under: {root_dir}")

        labels = sorted({sample["label"] for sample in self.samples})
        self.label_to_idx = {label: i for i, label in enumerate(labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}

    def _build_samples(self):
        """
        Walk through the directory tree and collect every valid mouths folder.
        Expected pattern:
            root_dir / subject_id / word_label / mouths / *.png
        """
        for subject in sorted(os.listdir(self.root_dir)):
            subject_path = os.path.join(self.root_dir, subject)
            if not os.path.isdir(subject_path):
                continue

            for word in sorted(os.listdir(subject_path)):
                word_path = os.path.join(subject_path, word)
                if not os.path.isdir(word_path):
                    continue

                mouths_path = os.path.join(word_path, "mouths")
                if not os.path.isdir(mouths_path):
                    continue

                pngs = [
                    f for f in os.listdir(mouths_path)
                    if f.lower().endswith(".png")
                ]
                if len(pngs) == 0:
                    continue

                self.samples.append({
                    "frames_dir": mouths_path,
                    "label": word,
                    "subject_id": subject,
                    "num_frames_total": len(pngs),
                })

    def __len__(self):
        return len(self.samples)

    def _get_sorted_pngs(self, frames_dir):
        pngs = [
            os.path.join(frames_dir, f)
            for f in os.listdir(frames_dir)
            if f.lower().endswith(".png")
        ]
        pngs = sorted(pngs)
        return pngs

    def _sample_indices(self, n_total):
        if n_total >= self.num_frames:
            indices = torch.linspace(0, n_total - 1, steps=self.num_frames).long().tolist()
        else:
            indices = list(range(n_total))
            while len(indices) < self.num_frames:
                indices.append(n_total - 1)
        return indices

    def _load_frame(self, path):
        img = Image.open(path)

        if self.grayscale:
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        img = img.resize((self.image_size, self.image_size))

        img = torch.tensor(list(img.getdata()), dtype=torch.float32)

        if self.grayscale:
            img = img.view(self.image_size, self.image_size) / 255.0
        else:
            img = img.view(self.image_size, self.image_size, 3) / 255.0
            img = img.permute(2, 0, 1)  # (3, H, W)

        return img

    def __getitem__(self, idx):
        sample = self.samples[idx]
        frames_dir = sample["frames_dir"]
        label = sample["label"]

        png_paths = self._get_sorted_pngs(frames_dir)
        if len(png_paths) == 0:
            raise ValueError(f"No PNGs found in {frames_dir}")

        indices = self._sample_indices(len(png_paths))
        selected_paths = [png_paths[i] for i in indices]

        frames = [self._load_frame(p) for p in selected_paths]

        if self.grayscale:
            # frames: list of (H, W)
            x = torch.stack(frames, dim=0)  # (T, H, W)
            x = x.unsqueeze(0)              # (1, T, H, W)
        else:
            # frames: list of (3, H, W)
            x = torch.stack(frames, dim=0)  # (T, 3, H, W)
            x = x.permute(1, 0, 2, 3)       # (3, T, H, W)

        y = self.label_to_idx[label]
        return x, y