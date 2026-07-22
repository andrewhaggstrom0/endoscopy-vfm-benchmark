"""CLIP vision encoder wrapper.

The deliberate contrast to DINOv2. CLIP is trained to align images with text
captions, not to build spatially coherent dense features -- so its patch tokens
are weak and its signal concentrates in the pooled representation. This is
exactly why it is worth benchmarking: prediction P2 is that CLIP probes
respectably (semantic content is there) but is temporally *worse* than DINOv2
(caption-alignment has no reason to be frame-to-frame stable).

Two encoder-specific details that matter for a fair comparison:

1. CLIP has its OWN normalization stats, not ImageNet's. Feeding ImageNet-
   normalized pixels to CLIP is a silent ~few-point accuracy hit and a classic
   cross-model-comparison bug. The stats below are OpenAI CLIP's.

2. `pool="cls"` here returns the pooled output. HF's CLIPVisionModel exposes
   both last_hidden_state (patch + CLS tokens) and pooler_output (the CLS token
   after final layernorm). We use the post-layernorm pooled vector -- the
   representation CLIP was actually trained to make useful -- and expose raw
   patch tokens too so mean_patch remains available for ablations.

Patch size is 14 or 16 depending on the variant; the guard reads it from the
loaded config rather than assuming.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseEncoder, EncoderConfig

# OpenAI CLIP normalization -- NOT ImageNet.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

PRESETS = {
    "clip_vitb16": "openai/clip-vit-base-patch16",
    "clip_vitb32": "openai/clip-vit-base-patch32",
    "clip_vitl14": "openai/clip-vit-large-patch14",
}


class CLIPEncoder(BaseEncoder):
    def __init__(self, cfg: EncoderConfig, device: str = "cuda"):
        self._mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        super().__init__(cfg, device)
        # Guard after load, when patch size is known from the config.
        patch = self.model.config.patch_size
        if cfg.image_size % patch != 0:
            raise ValueError(
                f"CLIP patch size is {patch}; image_size={cfg.image_size} "
                f"is not a multiple."
            )

    def _load(self) -> torch.nn.Module:
        from transformers import CLIPVisionModel

        repo = PRESETS.get(self.cfg.weights, self.cfg.weights)
        return CLIPVisionModel.from_pretrained(repo)

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
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
        # pooler_output: (B, D) post-layernorm CLS -- what CLIP was trained on.
        # last_hidden_state[:, 1:]: patch tokens, for mean_patch ablations.
        return {"cls": out.pooler_output,
                "patches": out.last_hidden_state[:, 1:]}


def build(name: str = "clip_vitb16", device: str = "cuda", **kwargs) -> CLIPEncoder:
    cfg = EncoderConfig(name=name, weights=name, **kwargs)
    return CLIPEncoder(cfg, device=device)
