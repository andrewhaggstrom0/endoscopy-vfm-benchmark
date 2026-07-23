"""Optical-flow coupling: is embedding motion tracking scene motion, or noise?

Low inter-frame drift is only a virtue if the encoder moves WHEN THE SCENE
MOVES. An encoder whose embedding velocity is uncorrelated with actual pixel
motion is not stable -- it is jittering independently of the surgery, which is
a stronger indictment than a raw drift number.

We compute dense Farneback flow between consecutive 1fps frames, take mean
magnitude per pair, and correlate (Spearman, per video) against embedding
velocity from the same pair. Spearman rather than Pearson because the
relationship need not be linear -- only monotone.

Cost note: this re-reads frame pixels, so it is the expensive pass. Frames are
downscaled before flow; magnitude is a relative signal and full resolution buys
nothing for a rank correlation.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr


def flow_magnitude_sequence(frame_paths: list[Path],
                            resize_to: int = 256) -> np.ndarray:
    """Mean dense-flow magnitude between consecutive frames. Length T-1."""
    mags = []
    prev = None
    for p in frame_paths:
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"failed to read {p}")
        h, w = img.shape
        scale = resize_to / max(h, w)
        img = cv2.resize(img, (int(w * scale), int(h * scale)),
                         interpolation=cv2.INTER_AREA)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev, img, None,
                pyr_scale=0.5, levels=3, winsize=15, iterations=3,
                poly_n=5, poly_sigma=1.2, flags=0)
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mags.append(float(mag.mean()))
        prev = img
    return np.asarray(mags)


def velocity_flow_coupling(velocity: np.ndarray,
                           flow_mag: np.ndarray) -> dict:
    """Spearman coupling between embedding velocity and scene motion.

    rho near 1  -> drift is tracking the scene (signal)
    rho near 0  -> drift is independent of the scene (noise)
    """
    n = min(len(velocity), len(flow_mag))
    if n < 10:
        return {"rho": float("nan"), "p": float("nan"), "n": n}
    rho, p = spearmanr(velocity[:n], flow_mag[:n])
    return {"rho": float(rho), "p": float(p), "n": int(n)}
