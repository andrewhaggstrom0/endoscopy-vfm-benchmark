"""Cache-layer tests. No GPU, no dataset, no model -- run these anywhere."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.cache.store import EmbeddingStore

FP = {"encoder": "test", "config_hash": "abc123"}


def _store(tmp_path):
    return EmbeddingStore(tmp_path)


def _write(st, emb=None, ids=None):
    emb = np.random.rand(10, 8).astype(np.float32) if emb is None else emb
    ids = [f"f{i}" for i in range(len(emb))] if ids is None else ids
    return st.write(encoder="test", config_hash="abc123", dataset="d",
                    video_id="v1", embeddings=emb, frame_ids=ids, fingerprint=FP)


def test_roundtrip(tmp_path):
    st = _store(tmp_path)
    emb = np.random.rand(10, 8).astype(np.float32)
    _write(st, emb)
    got = st.read(encoder="test", config_hash="abc123", dataset="d", video_id="v1")
    np.testing.assert_array_equal(np.asarray(got.embeddings), emb)
    assert len(got) == 10
    assert got.meta["embed_dim"] == 8


def test_detects_corruption(tmp_path):
    """The whole reason this layer exists."""
    st = _store(tmp_path)
    npy = _write(st)
    arr = np.load(npy)
    arr[3, 2] += 0.001          # plausible value, no NaN, correct shape
    np.save(npy, arr)
    with pytest.raises(RuntimeError, match="CACHE CORRUPT"):
        st.read(encoder="test", config_hash="abc123", dataset="d", video_id="v1")


def test_rejects_non_finite(tmp_path):
    st = _store(tmp_path)
    emb = np.random.rand(4, 8).astype(np.float32)
    emb[1, 1] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        _write(st, emb)


def test_rejects_wrong_dtype(tmp_path):
    with pytest.raises(TypeError):
        _write(_store(tmp_path), np.random.rand(4, 8).astype(np.float64))


def test_rejects_id_count_mismatch(tmp_path):
    with pytest.raises(ValueError, match="frame_ids"):
        _write(_store(tmp_path), np.random.rand(4, 8).astype(np.float32), ["a", "b"])


def test_config_hash_partitions_storage(tmp_path):
    """Two settings must not collide."""
    st = _store(tmp_path)
    e = np.random.rand(4, 8).astype(np.float32)
    ids = ["a", "b", "c", "d"]
    for h in ("hash_A", "hash_B"):
        st.write(encoder="test", config_hash=h, dataset="d", video_id="v1",
                 embeddings=e, frame_ids=ids, fingerprint=FP)
    assert st.shard_dir("test", "hash_A", "d") != st.shard_dir("test", "hash_B", "d")
    assert st.list_videos("test", "hash_A", "d") == ["v1"]


def test_manifest_records_provenance(tmp_path):
    st = _store(tmp_path)
    npy = _write(st)
    meta = json.loads(npy.with_suffix(".json").read_text())
    for k in ("sha256", "git_commit", "written_utc", "fingerprint", "norm_mean"):
        assert k in meta


def test_verify_all(tmp_path):
    st = _store(tmp_path)
    _write(st)
    assert st.verify_all("test", "abc123", "d") == {"v1": "ok"}


def test_no_temp_files_left_behind(tmp_path):
    """Regression: np.save appends .npy to paths not ending in it, which made
    the atomic-rename write to v1.npy.tmp.npy and orphan a stray file."""
    st = _store(tmp_path)
    _write(st)
    strays = list(tmp_path.rglob("*.tmp*"))
    assert not strays, strays
    assert (tmp_path / "test" / "abc123" / "d" / "v1.npy").exists()
