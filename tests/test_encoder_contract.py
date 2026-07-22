"""Contract tests every encoder must pass.

Deliberately cheap: these run on random noise, need no dataset, and catch the
failure modes that are expensive to find later -- nondeterminism, dtype drift,
batch-dependence, and preprocessing that silently accepts the wrong input.

Run: pytest tests/test_encoder_contract.py -v
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from src.encoders.base import Encoder, EncoderConfig
from src.encoders.dinov2 import DINOv2Encoder

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture(scope="module")
def enc():
    cfg = EncoderConfig(name="dinov2_vits14", weights="dinov2_vits14",
                        image_size=224, pool="cls")
    return DINOv2Encoder(cfg, device=DEVICE)


def _frames(n=4, h=480, w=854, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, h, w, 3), dtype=np.uint8)


def test_satisfies_protocol(enc):
    assert isinstance(enc, Encoder)


def test_output_shape_and_dtype(enc):
    out = enc.encode(enc.preprocess(_frames()))
    assert out.shape == (4, enc.embed_dim)
    assert out.dtype == torch.float32
    assert out.device.type == "cpu"
    assert torch.isfinite(out).all()


def test_deterministic(enc):
    """Same input twice must give bit-identical output.

    Nondeterminism here would make every temporal-drift measurement in the
    project meaningless -- drift between adjacent frames is small, and encoder
    jitter of the same magnitude is indistinguishable from it.
    """
    x = enc.preprocess(_frames())
    a, b = enc.encode(x), enc.encode(x)
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_batch_invariance(enc):
    """Embedding of a frame must not depend on what else is in its batch."""
    f = _frames(n=4)
    full = enc.encode(enc.preprocess(f))
    solo = enc.encode(enc.preprocess(f[2:3]))
    torch.testing.assert_close(full[2], solo[0], rtol=1e-4, atol=1e-5)


def test_preprocess_rejects_bad_input(enc):
    with pytest.raises(TypeError):
        enc.preprocess(_frames().astype(np.float32))
    with pytest.raises(ValueError):
        enc.preprocess(_frames()[..., :1])


def test_patch_size_guard():
    with pytest.raises(ValueError, match="patch size"):
        DINOv2Encoder(EncoderConfig(name="x", weights="dinov2_vits14",
                                    image_size=256), device=DEVICE)


def test_fingerprint_is_stable_and_complete(enc):
    fp = enc.fingerprint()
    for key in ("encoder", "config_hash", "embed_dim", "weights_sha256"):
        assert key in fp and fp[key] is not None
    assert fp == enc.fingerprint()


def test_config_hash_changes_with_pool():
    a = EncoderConfig(name="d", weights="w", pool="cls")
    b = EncoderConfig(name="d", weights="w", pool="mean_patch")
    assert a.hash() != b.hash()


def test_identical_frames_give_zero_drift(enc):
    """Sanity control for the Week 3 temporal metrics.

    A static clip must produce ~zero inter-frame distance. If this fails, the
    drift measurements later are measuring the encoder, not the video.
    """
    f = np.repeat(_frames(n=1), 8, axis=0)
    e = enc.encode(enc.preprocess(f))
    d = 1 - torch.nn.functional.cosine_similarity(e[:-1], e[1:], dim=-1)
    assert d.abs().max() < 1e-5, d


# --- CLIP: same contract, different backbone -------------------------------

@pytest.fixture(scope="module")
def clip_enc():
    from src.encoders.clip import CLIPEncoder
    cfg = EncoderConfig(name="clip_vitb16", weights="clip_vitb16",
                        image_size=224, pool="cls")
    return CLIPEncoder(cfg, device=DEVICE)


def test_clip_satisfies_protocol(clip_enc):
    assert isinstance(clip_enc, Encoder)


def test_clip_deterministic(clip_enc):
    x = clip_enc.preprocess(_frames())
    assert torch.equal(clip_enc.encode(x), clip_enc.encode(x))


def test_clip_zero_drift_on_static_clip(clip_enc):
    f = np.repeat(_frames(n=1), 8, axis=0)
    e = clip_enc.encode(clip_enc.preprocess(f))
    d = 1 - torch.nn.functional.cosine_similarity(e[:-1], e[1:], dim=-1)
    assert d.abs().max() < 1e-5, d


def test_clip_uses_own_norm_not_imagenet(clip_enc):
    """Regression guard: CLIP must not silently inherit ImageNet stats."""
    from src.encoders.clip import CLIP_MEAN
    from src.encoders.dinov2 import IMAGENET_MEAN
    assert CLIP_MEAN != IMAGENET_MEAN
    assert torch.allclose(clip_enc._mean.flatten(),
                          torch.tensor(CLIP_MEAN), atol=1e-6)
