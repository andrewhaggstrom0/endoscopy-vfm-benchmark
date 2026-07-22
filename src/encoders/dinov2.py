"""DINOv2 wrapper.

The reference encoder for this benchmark: self-supervised on natural images,
no language supervision, and the strongest published frozen-feature baseline on
surgical phase recognition. Everything else is measured against it.

Preprocessing note
------------------
DINOv2 uses ImageNet normalization and a patch size of 14, so the input side
length must be a multiple of 14. 224 = 16 x 14 works; 256 does not, and passing
it produces a silent shape mismatch in the patch grid rather than an error. The
check below turns that into a loud failure at construction time.

Endoscopic frames are typically 854x480 or 1920x1080 with black pillarboxing.
Resizing the full frame squashes the circular scope view; the pillarbox is
handled in the dataset layer (`src/data/`), not here, so that every encoder
sees identically cropped pixels. Keeping the crop out of the encoder is what
makes cross-model comparison valid.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseEncoder, EncoderConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

PATCH = 14

PRESETS = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
}


class DINOv2Encoder(BaseEncoder):
    def __init__(self, cfg: EncoderConfig, device: str = "cuda"):
        if cfg.image_size % PATCH != 0:
            raise ValueError(
                f"DINOv2 patch size is {PATCH}; image_size={cfg.image_size} "
                f"is not a multiple. Use 224 (16x14) or 518 (37x14)."
            )
        self._mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        super().__init__(cfg, device)

    def _load(self) -> torch.nn.Module:
        from transformers import AutoModel

        repo = PRESETS.get(self.cfg.weights, self.cfg.weights)
        return AutoModel.from_pretrained(repo)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """(B, H, W, 3) uint8 RGB -> (B, 3, S, S) normalized float32.

        Bicubic + antialias, matching the DINOv2 eval protocol. Interpolation
        mode is not a free choice: bilinear-without-antialias on downsampled
        surgical frames measurably changes probe accuracy, and mismatching the
        original protocol is a classic way to fail a reproduction gate.
        """
        if frames.dtype != np.uint8:
            raise TypeError(f"expected uint8 frames, got {frames.dtype}")
        if frames.ndim != 4 or frames.shape[-1] != 3:
            raise ValueError(f"expected (B, H, W, 3), got {frames.shape}")

        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float().div_(255.0)
        s = self.cfg.image_size
        x = F.interpolate(x, size=(s, s), mode="bicubic",
                          align_corners=False, antialias=True)
        x = x.clamp_(0, 1)
        return (x - self._mean) / self._std

    def _forward_features(self, batch: torch.Tensor) -> dict:
        out = self.model(pixel_values=batch)
        h = out.last_hidden_state          # (B, 1 + N, D)
        return {"cls": h[:, 0], "patches": h[:, 1:]}


def build(name: str = "dinov2_vitl14", device: str = "cuda", **kwargs) -> DINOv2Encoder:
    """Factory used by configs. Keeps YAML free of import paths."""
    cfg = EncoderConfig(name=name, weights=name, **kwargs)
    return DINOv2Encoder(cfg, device=device)
