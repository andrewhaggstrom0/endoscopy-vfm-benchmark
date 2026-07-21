"""Extraction entrypoint: config -> verified shards on disk.

    python -m src.extract --config configs/dinov2_cholecseg8k.yaml
    python -m src.extract --config ... --limit-videos 2   # smoke test
    python -m src.extract --config ... --verify-only

Idempotent: existing shards are skipped unless --overwrite. A job that dies
halfway can be resubmitted without redoing completed videos.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from src.cache.store import EmbeddingStore
from src.data import cholecseg8k
from src.encoders.dinov2 import build as build_dinov2

BUILDERS = {"src.encoders.dinov2.build": build_dinov2}
LOADERS = {"cholecseg8k": cholecseg8k}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit-videos", type=int, default=None)
    ap.add_argument("--limit-frames", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    OmegaConf.resolve(cfg)

    enc_cfg = OmegaConf.to_container(cfg.encoder, resolve=True)
    target = enc_cfg.pop("_target_")
    encoder = BUILDERS[target](**enc_cfg)
    encoder.device = args.device

    fingerprint = encoder.fingerprint()
    chash = encoder.cfg.hash()
    dataset = cfg.data.dataset
    store = EmbeddingStore(cfg.extract.out_dir)

    print(f"encoder      {encoder.cfg.name}  dim={encoder.embed_dim}")
    print(f"config_hash  {chash}")
    print(f"weights_sha  {fingerprint['weights_sha256'][:16]}...")
    print(f"out          {store.shard_dir(encoder.cfg.name, chash, dataset)}")

    if args.verify_only:
        results = store.verify_all(encoder.cfg.name, chash, dataset)
        bad = {k: v for k, v in results.items() if v != "ok"}
        print(f"\nverified {len(results)} shards, {len(bad)} bad")
        for k, v in bad.items():
            print(f"  {k}: {v}")
        return 1 if bad else 0

    loader = LOADERS[dataset]
    groups = loader.discover(cfg.data.root)
    vids = list(groups)[: args.limit_videos]
    print(f"videos       {len(vids)} of {len(groups)}\n")

    bs = int(cfg.extract.batch_size)
    total_frames = 0
    t_start = time.time()

    for i, vid in enumerate(vids, 1):
        if store.exists(encoder.cfg.name, chash, dataset, vid) and not args.overwrite:
            print(f"[{i}/{len(vids)}] {vid}: exists, skipping")
            continue

        paths = groups[vid][: args.limit_frames]
        embs, t0 = [], time.time()
        for s in range(0, len(paths), bs):
            frames = loader.load_frames(paths[s : s + bs])
            embs.append(encoder.encode(encoder.preprocess(frames)).numpy())
        emb = np.concatenate(embs).astype(np.float32)
        dt = time.time() - t0

        store.write(
            encoder=encoder.cfg.name, config_hash=chash, dataset=dataset,
            video_id=vid, embeddings=emb,
            frame_ids=[p.stem for p in paths],
            fingerprint=fingerprint,
            extra={"extract_seconds": round(dt, 2),
                   "fps": round(len(paths) / dt, 1),
                   "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1)
                                   if args.device == "cuda" else None},
        )
        total_frames += len(paths)
        print(f"[{i}/{len(vids)}] {vid}: {emb.shape} in {dt:.1f}s "
              f"({len(paths)/dt:.1f} fps)")

    elapsed = time.time() - t_start
    print(f"\n{total_frames} frames in {elapsed:.1f}s")
    print("verify with: --verify-only")
    return 0


if __name__ == "__main__":
    sys.exit(main())
