"""Is CLIP's low drift real stability, or just low feature variance?

For each encoder/video: step-to-step drift vs the video's overall embedding
spread. A truly stable encoder has drift SMALL relative to its spread. A bland
encoder has small drift because everything is bland -- the ratio exposes it.
"""
from __future__ import annotations
import os, glob, json
import numpy as np, pandas as pd

frames = f"{os.environ['BIGDIR']}/endoscopy/raw/cholec80_frames"

def stats(cache_dir, name):
    rows = []
    for npy in sorted(glob.glob(f"{cache_dir}/*.npy")):
        E = np.load(npy)
        En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-8)
        # step drift
        step = 1 - np.sum(En[:-1] * En[1:], axis=1)
        # overall spread: mean pairwise cosine distance to the video centroid
        c = En.mean(0); c /= (np.linalg.norm(c) + 1e-8)
        spread = 1 - (En @ c)
        rows.append({
            "encoder": name,
            "drift": step.mean(),
            "spread": spread.mean(),
            "drift_over_spread": step.mean() / (spread.mean() + 1e-8),
            "emb_std": E.std(),
        })
    return pd.DataFrame(rows)

d = pd.concat([
    stats(os.path.expandvars(p), n) for n, p in [
        ("dinov2_vits14", glob.glob(f"{os.environ['BIGDIR']}/endoscopy/cache/dinov2_vits14/*/cholec80")[0]),
        ("clip_vitb16",   glob.glob(f"{os.environ['BIGDIR']}/endoscopy/cache/clip_vitb16/*/cholec80")[0]),
    ]], ignore_index=True)

print(d.groupby("encoder")[["drift","spread","drift_over_spread","emb_std"]]
        .mean().round(4).to_string())
print("\nread: if CLIP's spread and emb_std are much smaller, its low drift is")
print("low variance, not stability. Compare drift_over_spread -- that's the")
print("variance-normalized number. If CLIP no longer wins on it, the inversion")
print("is a metric artifact, not a finding.")
