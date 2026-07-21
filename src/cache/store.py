"""Memory-mapped embedding store with verified manifests.

Layout
------
    {root}/{encoder}/{config_hash}/{dataset}/{video_id}.npy
    {root}/{encoder}/{config_hash}/{dataset}/{video_id}.json

The config hash sits in the *path*, not just the manifest. Changing pooling or
image size therefore writes to a new directory instead of overwriting -- two
extraction settings can never be silently mixed inside one analysis.

Every read verifies the SHA256 recorded at write time. This is deliberately
paranoid: a corrupted shard that is statistically plausible (right shape, no
NaNs, sane norms) is undetectable downstream and can cost days. Hashing at read
time converts that class of bug into an immediate, loud failure.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


@dataclass
class Shard:
    embeddings: np.ndarray      # (N, D) float32, memory-mapped
    frame_ids: list[str]
    meta: dict

    def __len__(self) -> int:
        return len(self.frame_ids)


class EmbeddingStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def shard_dir(self, encoder: str, config_hash: str, dataset: str) -> Path:
        return self.root / encoder / config_hash / dataset

    def _paths(self, encoder, config_hash, dataset, video_id):
        d = self.shard_dir(encoder, config_hash, dataset)
        return d / f"{video_id}.npy", d / f"{video_id}.json"

    def exists(self, encoder, config_hash, dataset, video_id) -> bool:
        npy, js = self._paths(encoder, config_hash, dataset, video_id)
        return npy.exists() and js.exists()

    def write(self, *, encoder: str, config_hash: str, dataset: str,
              video_id: str, embeddings: np.ndarray, frame_ids: list[str],
              fingerprint: dict, extra: dict | None = None) -> Path:
        if embeddings.dtype != np.float32:
            raise TypeError(f"expected float32, got {embeddings.dtype}")
        if embeddings.ndim != 2:
            raise ValueError(f"expected (N, D), got {embeddings.shape}")
        if len(frame_ids) != embeddings.shape[0]:
            raise ValueError(
                f"{len(frame_ids)} frame_ids vs {embeddings.shape[0]} rows"
            )
        if not np.isfinite(embeddings).all():
            bad = int((~np.isfinite(embeddings)).sum())
            raise ValueError(f"{bad} non-finite values in {video_id}")

        npy, js = self._paths(encoder, config_hash, dataset, video_id)
        npy.parent.mkdir(parents=True, exist_ok=True)

        # Write to a temp file then rename. Rename is atomic on POSIX, so a
        # job killed mid-write leaves no half-shard that looks complete.
        # np.save() appends ".npy" to any path not already ending in it, so
        # np.save("v1.npy.tmp") silently writes "v1.npy.tmp.npy". Passing an
        # open file handle bypasses that entirely.
        tmp = npy.with_name(npy.name + ".tmp")
        with open(tmp, "wb") as f:
            np.save(f, embeddings, allow_pickle=False)
        tmp.replace(npy)

        meta = {
            "video_id": video_id,
            "dataset": dataset,
            "n_frames": int(embeddings.shape[0]),
            "embed_dim": int(embeddings.shape[1]),
            "frame_ids": frame_ids,
            "sha256": sha256_file(npy),
            "fingerprint": fingerprint,
            "git_commit": git_commit(),
            "written_utc": datetime.now(timezone.utc).isoformat(),
            "norm_mean": float(np.linalg.norm(embeddings, axis=1).mean()),
            "norm_std": float(np.linalg.norm(embeddings, axis=1).std()),
            **(extra or {}),
        }
        js.write_text(json.dumps(meta, indent=2))
        return npy

    def read(self, *, encoder: str, config_hash: str, dataset: str,
             video_id: str, verify: bool = True) -> Shard:
        npy, js = self._paths(encoder, config_hash, dataset, video_id)
        if not npy.exists():
            raise FileNotFoundError(npy)
        meta = json.loads(js.read_text())
        if verify:
            actual = sha256_file(npy)
            if actual != meta["sha256"]:
                raise RuntimeError(
                    f"CACHE CORRUPT: {npy}\n"
                    f"  manifest sha256 {meta['sha256']}\n"
                    f"  actual   sha256 {actual}\n"
                    f"  written  {meta['written_utc']} @ {meta['git_commit'][:8]}\n"
                    f"Delete the shard and re-extract; do not use these values."
                )
        arr = np.load(npy, mmap_mode="r")
        return Shard(embeddings=arr, frame_ids=meta["frame_ids"], meta=meta)

    def list_videos(self, encoder, config_hash, dataset) -> list[str]:
        d = self.shard_dir(encoder, config_hash, dataset)
        return sorted(p.stem for p in d.glob("*.npy")) if d.exists() else []

    def verify_all(self, encoder, config_hash, dataset) -> dict[str, str]:
        """Sweep every shard. Returns {video_id: "ok" | error message}."""
        out = {}
        for vid in self.list_videos(encoder, config_hash, dataset):
            try:
                self.read(encoder=encoder, config_hash=config_hash,
                          dataset=dataset, video_id=vid, verify=True)
                out[vid] = "ok"
            except Exception as e:
                out[vid] = str(e)
        return out
