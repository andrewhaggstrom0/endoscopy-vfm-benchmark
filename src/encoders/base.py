"""Uniform interface for every vision foundation model in the benchmark.

Design contract
---------------
Every experiment downstream of extraction is a pure function over cached
embeddings. That only holds if extraction is deterministic and fully described
by metadata. So each encoder must:

  1. Be frozen and in eval mode. No encoder here is ever trained.
  2. Expose `fingerprint()` -- everything that could change the numbers.
     Hashed into the cache manifest. If any field changes, the cache is stale.
     This makes "did I compare the same thing across weeks?" answerable rather
     than hopeful.
  3. Own its own preprocessing. Normalization stats and resize interpolation
     differ per model, and a mismatch here is the single most common cause of
     a linear probe landing well below published baselines.

Pooling
-------
`pool` is a first-class config field, not an implementation detail. CLS-token
and mean-patch pooling produce measurably different temporal stability from the
*same* backbone -- exactly the axis this project measures. Held fixed across
models by default; varied only in a labeled ablation.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Protocol, runtime_checkable

import numpy as np
import torch

PoolMode = Literal["cls", "mean_patch", "cls_plus_mean"]


@dataclass(frozen=True)
class EncoderConfig:
    """Everything that affects the output embeddings.

    Frozen because it is hashed -- a config mutated after extraction would
    silently invalidate the manifest.
    """

    name: str                      # short id, e.g. "dinov2_vitl14"
    weights: str                   # HF repo id or checkpoint path
    image_size: int = 224
    pool: PoolMode = "cls"
    dtype: Literal["fp32", "fp16", "bf16"] = "fp32"
    l2_normalize: bool = False
    extra: dict = field(default_factory=dict)

    def hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@runtime_checkable
class Encoder(Protocol):
    """Structural interface. Wrappers need not subclass anything."""

    cfg: EncoderConfig
    embed_dim: int

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        """(B, H, W, 3) uint8 RGB -> (B, 3, h, w) float model-space tensor."""
        ...

    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        """(B, 3, h, w) -> (B, D) float32 on CPU, gradient-free."""
        ...

    def fingerprint(self) -> dict:
        """Metadata written into the cache manifest."""
        ...


class BaseEncoder:
    """Shared machinery. Subclasses implement `_load` and `_forward_features`."""

    def __init__(self, cfg: EncoderConfig, device: str = "cuda"):
        self.cfg = cfg
        self.device = device
        self.model = self._load()
        self.model.eval().to(device)
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.embed_dim = self._infer_embed_dim()

    # ---- subclass hooks -------------------------------------------------

    def _load(self) -> torch.nn.Module:
        raise NotImplementedError

    def _forward_features(self, batch: torch.Tensor) -> dict:
        """Return {"cls": (B, D) or None, "patches": (B, N, D) or None}."""
        raise NotImplementedError

    def preprocess(self, frames: np.ndarray) -> torch.Tensor:
        raise NotImplementedError

    # ---- shared ---------------------------------------------------------

    @property
    def _torch_dtype(self) -> torch.dtype:
        return {"fp32": torch.float32,
                "fp16": torch.float16,
                "bf16": torch.bfloat16}[self.cfg.dtype]

    def _pool(self, feats: dict) -> torch.Tensor:
        cls, patches = feats.get("cls"), feats.get("patches")
        mode = self.cfg.pool
        if mode == "cls":
            if cls is None:
                raise ValueError(f"{self.cfg.name} exposes no CLS token")
            return cls
        if mode == "mean_patch":
            if patches is None:
                raise ValueError(f"{self.cfg.name} exposes no patch tokens")
            return patches.mean(dim=1)
        if mode == "cls_plus_mean":
            if cls is None or patches is None:
                raise ValueError(f"{self.cfg.name} lacks CLS or patch tokens")
            return torch.cat([cls, patches.mean(dim=1)], dim=-1)
        raise ValueError(f"unknown pool mode: {mode}")

    @torch.inference_mode()
    def encode(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device, dtype=self._torch_dtype, non_blocking=True)
        emb = self._pool(self._forward_features(batch))
        # Cast to fp32 *before* leaving the GPU. Cosine distances between
        # near-identical adjacent frames are the core measurement in this
        # project, and fp16 rounding error is the same order as the signal.
        emb = emb.float()
        if self.cfg.l2_normalize:
            # Off by default: normalizing discards magnitude, which is itself
            # part of the temporal-drift signal. Enable only for retrieval-
            # style experiments -- it is in the config hash, so it is recorded.
            emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.cpu()

    def _infer_embed_dim(self) -> int:
        dummy = torch.zeros(1, 3, self.cfg.image_size, self.cfg.image_size)
        return int(self.encode(dummy).shape[-1])

    def fingerprint(self) -> dict:
        return {
            "encoder": self.cfg.name,
            "config": asdict(self.cfg),
            "config_hash": self.cfg.hash(),
            "embed_dim": self.embed_dim,
            "weights_sha256": self._weights_sha256(),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
        }

    def _weights_sha256(self) -> str:
        """Hash of the loaded state dict -- catches a silently swapped checkpoint.

        Runs once per extraction job. This is the check that turns "the weights
        were probably the same" into a fact you can grep for in a manifest.
        """
        h = hashlib.sha256()
        for k, v in sorted(self.model.state_dict().items()):
            h.update(k.encode())
            h.update(v.detach().cpu().contiguous().float().numpy().tobytes())
        return h.hexdigest()
