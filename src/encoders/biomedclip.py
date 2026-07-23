"""BiomedCLIP encoder -- medical-domain, language-supervised.

Completes the pretraining 2x2:

                    general              medical/surgical
  self-supervised   DINOv2               EndoViT
  language-sup.     CLIP                 BiomedCLIP

BiomedCLIP is trained on PMC-15M (biomedical figure-caption pairs from PubMed
Central). Note the distribution gap that matters for this benchmark: those are
mostly static figures -- histology, radiology, diagrams -- not endoscopic
video. So it is medical but not surgical, and not temporal. Whether that
domain proximity helps on laparoscopic frames is exactly the open question.

Implementation notes
--------------------
1. Ships via open_clip, not HF transformers. The vision tower is a timm ViT
   inside an open_clip TimmModel wrapper (`.trunk` = ViT, `.head` = projection).

2. Normalization stats are READ FROM the preprocessing transform open_clip
   returns, not hardcoded. Guessing them wrong is a silent accuracy hit, and
   BiomedCLIP does not necessarily use OpenAI CLIP's stats.

3. For comparability with src/encoders/clip.py we take the PRE-projection
   pooled representation (trunk output), not the projected image embedding.
   Both wrappers therefore probe the same kind of feature.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base import BaseEncoder, EncoderConfig

HF_HUB = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def _stats_from_transform(preprocess) -> tuple[tuple, tuple]:
    """Pull (mean, std) out of the torchvision Compose open_clip hands back."""
    for t in getattr(preprocess, "transforms", []):
        if hasattr(t, "mean") and hasattr(t, "std"):
            return tuple(float(x) for x in t.mean), tuple(float(x) for x in t.std)
    raise RuntimeError("could not locate Normalize in the open_clip transform; "
                       "inspect preprocess.transforms and set stats explicitly")


class BiomedCLIPEncoder(BaseEncoder):
    def __init__(self, cfg: EncoderConfig, device: str = "cuda"):
        self._mean = None      # populated in _load, from the real transform
        self._std = None
        super().__init__(cfg, device)
        if cfg.image_size % 16 != 0:
            raise ValueError(f"BiomedCLIP ViT-B/16; image_size={cfg.image_size} "
                             f"is not a multiple of 16.")

    def _load(self) -> torch.nn.Module:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(HF_HUB)
        mean, std = _stats_from_transform(preprocess)
        self._mean = torch.tensor(mean).view(1, 3, 1, 1)
        self._std = torch.tensor(std).view(1, 3, 1, 1)
        print(f"[biomedclip] normalization from transform: "
              f"mean={tuple(round(m,4) for m in mean)} "
              f"std={tuple(round(s,4) for s in std)}")

        visual = model.visual
        # Prefer the timm trunk (pre-projection), matching the CLIP wrapper.
        trunk = getattr(visual, "trunk", None)
        if trunk is not None:
            self._mode = "trunk"
            print("[biomedclip] using visual.trunk (pre-projection pooled)")
            return trunk
        self._mode = "visual"
        print("[biomedclip] WARNING: no .trunk; using full visual tower "
              "(projected features -- note the asymmetry vs CLIP wrapper)")
        return visual

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
        if self._mode == "trunk" and hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(batch)   # (B, 1+N, D)
            if feats.ndim == 3:
                return {"cls": feats[:, 0], "patches": feats[:, 1:]}
            return {"cls": feats, "patches": None}
        out = self.model(batch)
        return {"cls": out, "patches": None}


def build(name: str = "biomedclip_vitb16", device: str = "cuda",
          **kwargs) -> BiomedCLIPEncoder:
    cfg = EncoderConfig(name=name, weights=HF_HUB, **kwargs)
    return BiomedCLIPEncoder(cfg, device=device)
