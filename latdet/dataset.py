"""
Preloaded YOLO dataset — all data lives on GPU.

Loads images and labels once, preprocesses to GPU tensors,
then training iterates with zero CPU-GPU transfer.

CPU RAM is freed immediately after GPU transfer (image_list, label_list
are deleted and garbage-collected to avoid holding ~3 GB of CPU buffers).
"""
import gc
from pathlib import Path

import cv2
import numpy as np
import torch


class GPUPreloadedDataset:
    """
    Entire dataset pre-loaded into GPU memory as a flat tensor.

    Images:  (N, 3, img_size, img_size) float32 in [0, 1]  — on GPU only
    Labels:  list of (n_i, 5) tensors — [cls_id, cx, cy, w, h]  — on GPU only
    """

    def __init__(self, img_dir, label_dir, img_size=320, device='cuda', normalize=False):
        self.img_size = img_size
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.normalize = normalize

        img_dir = Path(img_dir)
        label_dir = Path(label_dir)
        img_paths = sorted(img_dir.glob('*.jpg')) + sorted(img_dir.glob('*.png'))

        pairs = []
        for p in img_paths:
            lp = label_dir / f'{p.stem}.txt'
            if lp.exists():
                pairs.append((p, lp))

        if not pairs:
            raise FileNotFoundError(f'No image-label pairs in {img_dir} / {label_dir}')

        print(f'Preloading {len(pairs)} samples to {self.device}...')

        image_list = []
        label_list = []
        total_boxes = 0
        dtype = torch.float16 if self.device.type == 'cuda' else torch.float32

        for img_path, label_path in pairs:
            # Read image
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
            img = img.astype(np.float32) / 255.0

            # Per-image contrast stretch for extremely dark images
            if self.normalize:
                low = np.percentile(img, 2)
                high = np.percentile(img, 98)
                if high > low:
                    img = (img - low) / (high - low)
                img = img.clip(0, 1)
            tensor = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W) CPU float32
            image_list.append(tensor)

            # Read labels
            boxes = []
            with open(label_path) as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        boxes.append([float(parts[0]), *map(float, parts[1:5])])
            lbl = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 5))
            label_list.append(lbl)
            total_boxes += len(boxes)

        # ── Stack and move to GPU, then free CPU buffers ──
        self.images = torch.stack(image_list, dim=0).to(self.device)  # (N, 3, H, W)
        self.labels = [lbl.to(self.device) for lbl in label_list]

        # Explicitly free CPU-side copies
        del image_list, label_list, pairs, img_paths
        gc.collect()

        self.n = len(self.labels)

        gb = self.images.element_size() * self.images.numel() / 1073741824
        print(f'  Images: {self.images.shape}  ({gb:.2f} GB on {self.device})')
        print(f'  Labels: {self.n} samples, {total_boxes} total boxes')
        print(f'  CPU buffers freed.')

    def __len__(self):
        return self.n


# ── GPU Augmentations ─────────────────────────────────────────────────

def augment_gpu(imgs, targets, img_size=320):
    """
    Batched GPU augmentations on pre-loaded tensors.

    Args:
        imgs:    (B, 3, H, W) float32 on GPU
        targets: list of B tensors each (n_i, 5)
    Returns:
        augmented images, augmented targets (same structure)
    """
    B = imgs.shape[0]
    device = imgs.device

    # Random horizontal flip (per-sample in batch)
    flip_mask = torch.rand(B, device=device) > 0.5
    if flip_mask.any():
        # Flip images: reverse width dimension
        imgs[flip_mask] = imgs[flip_mask].flip(dims=[-1])
        # Flip label cx
        for i in range(B):
            if flip_mask[i] and targets[i].numel() > 0:
                targets[i][:, 1] = 1.0 - targets[i][:, 1]

    # HSV jitter: simple brightness/contrast/saturation on GPU
    # Brightness
    bright = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4
    imgs = (imgs * bright).clamp(0, 1)

    # Contrast
    contrast = 1.0 + (torch.rand(B, 1, 1, 1, device=device) - 0.5) * 0.4
    mean = imgs.mean(dim=[2, 3], keepdim=True)
    imgs = ((imgs - mean) * contrast + mean).clamp(0, 1)

    return imgs, targets


# ── Collate (trivial — data is already on GPU) ────────────────────────

def gpu_collate(batch):
    """batch is already a tuple of (image, label) on GPU — just stack."""
    imgs, labels = zip(*batch)
    # imgs are individual tensors, labels are already tensors
    return torch.stack(imgs, 0), list(labels)
