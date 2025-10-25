from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, List, Tuple

import numpy as np
import torch


@dataclass
class ArtifactConfig:
    prob: float = 0.5
    # noise
    gaussian_noise_std: float = 0.05
    rician_noise_std: float = 0.0
    # intensity bias (low-frequency multiplicative field)
    bias_strength: float = 0.2
    # motion (small affine jitter)
    rot_deg: float = 5.0
    trans_pix: float = 5.0


def apply_gaussian_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return img
    noise = torch.randn_like(img) * std
    return img + noise


def apply_rician_noise(img: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return img
    n1 = torch.randn_like(img) * std
    n2 = torch.randn_like(img) * std
    return torch.sqrt((img + n1) ** 2 + n2 ** 2)


def apply_bias_field(img: torch.Tensor, strength: float) -> torch.Tensor:
    if strength <= 0:
        return img
    # generate smooth field using low-res noise upsampled
    n, c, h, w = 1, 1, img.shape[-2], img.shape[-1]
    low = torch.randn(1, 1, max(4, h // 32), max(4, w // 32), device=img.device)
    field = torch.nn.functional.interpolate(low, size=(h, w), mode="bilinear", align_corners=False)
    field = field.squeeze(0)
    field = (field - field.mean()) / (field.std() + 1e-6)
    field = 1.0 + strength * field
    return img * field


def apply_affine(img: torch.Tensor, lbl: torch.Tensor, rot_deg: float, trans_pix: float) -> Tuple[torch.Tensor, torch.Tensor]:
    if rot_deg <= 0 and trans_pix <= 0:
        return img, lbl
    angle = random.uniform(-rot_deg, rot_deg)
    tx = random.uniform(-trans_pix, trans_pix)
    ty = random.uniform(-trans_pix, trans_pix)
    grid = torch.nn.functional.affine_grid(
        torch.tensor([[
            [np.cos(np.deg2rad(angle)), -np.sin(np.deg2rad(angle)), tx / (img.shape[-1] / 2)],
            [np.sin(np.deg2rad(angle)),  np.cos(np.deg2rad(angle)), ty / (img.shape[-2] / 2)],
        ]], dtype=torch.float32, device=img.device),
        size=img.size(),
        align_corners=False,
    )
    img_w = torch.nn.functional.grid_sample(img, grid, mode="bilinear", align_corners=False)
    # labels: nearest
    lbl_f = lbl.float()
    lbl_w = torch.nn.functional.grid_sample(lbl_f, grid, mode="nearest", align_corners=False)
    return img_w, lbl_w.round().to(lbl.dtype)


class ArtifactAugmentDataset(torch.utils.data.Dataset):
    """Wrap a (img,lbl) dataset and apply artifacts on-the-fly for robustness evaluation/training.

    Expects dataset that returns (C,H,W) image tensor and (H,W) label tensor.
    """

    def __init__(self, base_ds: torch.utils.data.Dataset, cfg: ArtifactConfig):
        self.base = base_ds
        self.cfg = cfg

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int):
        img, lbl = self.base[idx]
        # to BCHW for warp functions
        img_b = img.unsqueeze(0)
        lbl_b = lbl.unsqueeze(0).unsqueeze(0)

        if random.random() < self.cfg.prob:
            # affine (motion)
            img_b, lbl_b = apply_affine(img_b, lbl_b, self.cfg.rot_deg, self.cfg.trans_pix)
            # noise
            img_b = apply_gaussian_noise(img_b, self.cfg.gaussian_noise_std)
            img_b = apply_rician_noise(img_b, self.cfg.rician_noise_std)
            # bias field
            img_b = apply_bias_field(img_b, self.cfg.bias_strength)

        img = img_b.squeeze(0)
        lbl = lbl_b.squeeze(0).squeeze(0)
        return img, lbl
