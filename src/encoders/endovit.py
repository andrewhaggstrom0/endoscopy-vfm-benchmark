"""EndoViT encoder -- the surgical-domain contrast.

EndoViT (Batic et al., 2024) is a ViT-B pretrained with a Masked Autoencoder
objective on Endo700k, ~700k endoscopic images from nine MIS datasets. It is
the domain-specific point in the benchmark: the model most likely to break the
DINOv2/CLIP pattern, because it has actually seen surgical scene dynamics.

Three differences from the other wrappers, all load-bearing:

1. Not an HF AutoModel. The published checkpoint is a plain timm
   VisionTransformer state_dict under key 'model'. We build the architecture
   and load weights by hand, following the official HF model card.

2. Its OWN normalization stats, computed on endoscopic images -- neither
   ImageNet (DINOv2) nor CLIP's. Wrong stats = silent accuracy loss.

3. MAE-pretrained, used with no masking at inference. Exposes CLS and patch
   tokens; MAE features are commonly mean-pooled patches, so both pools work.

DISCLOSURE for the report: Endo700k includes Cholec80, so EndoViT has seen
these videos during self-supervised pretraining (the authors excluded
downstream *test* sets only from backbone *finetuning*). EndoViT therefore has
a domain-familiarity advantage the general encoders lack. State this; do not
present it as a clean apples-to-apples win.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseEncoder, EncoderConfig

# EndoViT normalization -- endoscopic image stats (from the official model card).
ENDO_MEAN = (0.3464, 0.2280, 0.2228)
ENDO_STD = (0.2520, 0.2128, 0.2093)

HF_REPO = "egeozsoy/EndoViT"
WEIGHT_FILE = "pytorch_model.bin"   # verified at load; overridable via cfg.extra


class EndoViTEncoder(BaseEncoder):
    def __init__(self, cfg: EncoderConfig, device: str = "cuda"):
        self._mean = torch.tensor(ENDO_MEAN).view(1, 3, 1, 1)
        self._std = torch.tensor(ENDO_STD).view(1, 3, 1, 1)
        super().__init__(cfg, device)
        if cfg.image_size % 16 != 0:
            raise ValueError(f"EndoViT ViT-B patch 16; image_size={cfg.image_size} "
                             f"not a multiple of 16.")

    def _load(self) -> torch.nn.Module:
        from pathlib import Path

        from huggingface_hub import snapshot_download
        from timm.models.vision_transformer import VisionTransformer

        # ViT-B/16 architecture matching the EndoViT checkpoint.
        model = VisionTransformer(
            img_size=224, patch_size=16, embed_dim=768, depth=12,
            num_heads=12, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
            num_classes=0,   # feature extractor, no classification head
        )
        local = snapshot_download(repo_id=HF_REPO, revision="main")
        wfile = self.cfg.extra.get("weight_file", WEIGHT_FILE)
        ckpt_path = Path(local) / wfile
        if not ckpt_path.exists():
            # fall back to whatever single .bin/.pth the repo ships
            cands = list(Path(local).glob("*.bin")) + list(Path(local).glob("*.pth"))
            if len(cands) != 1:
                raise FileNotFoundError(
                    f"could not locate EndoViT weights in {local}; "
                    f"found {[c.name for c in cands]}. Set cfg.extra.weight_file.")
            ckpt_path = cands[0]

        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state = state.get("model", state)          # checkpoint stores under 'model'
        missing, unexpected = model.load_state_dict(state, strict=False)
        # MAE decoder / mask-token keys are expected to be unexpected; encoder
        # keys must NOT be missing. Fail loud if the backbone didn't populate.
        enc_missing = [k for k in missing if not k.startswith(("head", "fc_norm"))]
        if enc_missing:
            raise RuntimeError(f"EndoViT backbone keys missing: {enc_missing[:8]}")
        print(f"[endovit] loaded {ckpt_path.name}: "
              f"{len(missing)} missing, {len(unexpected)} unexpected (decoder ok)")
        return model

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
        # timm ViT forward_features -> (B, 1+N, D): token 0 is CLS, rest patches.
        feats = self.model.forward_features(batch)
        return {"cls": feats[:, 0], "patches": feats[:, 1:]}


def build(name: str = "endovit_vitb16", device: str = "cuda", **kwargs) -> EndoViTEncoder:
    cfg = EncoderConfig(name=name, weights=HF_REPO, **kwargs)
    return EndoViTEncoder(cfg, device=device)
