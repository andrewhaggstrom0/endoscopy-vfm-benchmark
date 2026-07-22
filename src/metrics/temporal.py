"""Temporal reliability metrics over frozen embeddings.

The project's core instrument. Frame-level probe accuracy says nothing about
whether a model's features are STABLE across a video -- and a surgical system
consumes a trajectory, not i.i.d. frames. These metrics quantify that stability.

Contract
--------
Every function takes ONE contiguous sequence: E of shape (T, D), frames in
temporal order, no gaps. The caller must never concatenate videos before
measuring -- a video boundary would register as maximal drift. Aggregation
across videos happens outside, at the reporting layer.

The degeneracy trap
-------------------
A model that outputs a constant vector has perfect drift (zero) and perfect
jitter (zero) -- and is useless. So stability alone is not a virtue; it is only
meaningful CONDITIONED on the features still being discriminative. That is what
`conditional_stability` encodes, and why raw drift is never reported alone.
"""

from __future__ import annotations

import numpy as np


def _l2norm(E: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + eps)


def feature_drift(E: np.ndarray) -> dict:
    """Mean cosine distance between consecutive frames.

    0 = identical adjacent embeddings (a static scene should approach this).
    Larger = the representation moves more per frame. Reported as mean and the
    95th percentile, because a few large jumps (true scene changes) matter
    differently from uniform jitter.
    """
    if E.shape[0] < 2:
        return {"drift_mean": 0.0, "drift_p95": 0.0, "drift_std": 0.0}
    En = _l2norm(E)
    cos = np.sum(En[:-1] * En[1:], axis=1)          # cosine sim of neighbors
    dist = 1.0 - cos
    return {
        "drift_mean": float(dist.mean()),
        "drift_p95": float(np.percentile(dist, 95)),
        "drift_std": float(dist.std()),
    }


def normalized_drift(E: np.ndarray) -> dict:
    """Step-to-step drift divided by the sequence's overall feature spread.

    Guards the low-variance loophole: a bland encoder that packs every frame
    into a tiny region of embedding space has small raw drift but is NOT
    tracking the scene -- its steps are small only because everything is small.
    Dividing by spread (mean cosine distance to the video centroid) exposes
    that. Lower drift_ratio = genuinely smaller relative motion per frame.

    Discovered empirically: CLIP's raw drift (0.042) beat DINOv2's (0.124), but
    CLIP's feature spread was 3.5x smaller; normalized, DINOv2 leads (0.67 vs
    0.80). Raw drift is gameable by variance; this is not.
    """
    if E.shape[0] < 2:
        return {"drift_ratio": 0.0, "spread": 0.0}
    En = _l2norm(E)
    step = (1.0 - np.sum(En[:-1] * En[1:], axis=1)).mean()
    c = En.mean(0); c = c / (np.linalg.norm(c) + 1e-8)
    spread = (1.0 - En @ c).mean()
    # Degenerate case: a (near-)constant encoder has ~zero spread, making the
    # ratio 0/0 -- pure float noise. A constant encoder is exactly the useless
    # case we penalize, so treat undefined spread as maximally unstable
    # (drift_ratio = 1.0 -> stability = 0).
    if spread < 1e-4:
        return {"drift_ratio": 1.0, "spread": float(spread)}
    return {"drift_ratio": float(step / spread), "spread": float(spread)}


def embedding_velocity(E: np.ndarray) -> np.ndarray:
    """Per-frame speed through embedding space: ||e_t - e_{t-1}|| on L2-normed
    features. Length T-1. Used by the optical-flow correlation (second pass)
    and as a raw trajectory-smoothness signal."""
    if E.shape[0] < 2:
        return np.zeros(0)
    En = _l2norm(E)
    return np.linalg.norm(En[1:] - En[:-1], axis=1)


def boundary_jitter(preds: np.ndarray) -> dict:
    """Predicted phase-transition rate from a per-frame classifier.

    A frozen frame-level probe with unstable features flip-flops between phases
    frame to frame, producing far more transitions than the surgery contains.
    We report transitions per 100 frames; the caller compares it to the
    ground-truth transition rate. A model at 90% accuracy emitting 40x the true
    transition count is the project's showcase failure.
    """
    if len(preds) < 2:
        return {"transitions": 0, "transitions_per_100": 0.0}
    changes = int(np.sum(preds[1:] != preds[:-1]))
    return {
        "transitions": changes,
        "transitions_per_100": float(100.0 * changes / (len(preds) - 1)),
    }


def phase_fragmentation(preds: np.ndarray) -> dict:
    """How chopped-up the prediction stream is.

    n_segments = count of maximal constant runs. A clean prediction of a
    7-phase surgery has ~7 segments; a jittery one has hundreds. mean_run_len
    is the average frames-per-segment -- short runs mean the model cannot hold
    a phase.
    """
    if len(preds) == 0:
        return {"n_segments": 0, "mean_run_len": 0.0}
    boundaries = np.where(preds[1:] != preds[:-1])[0]
    n_segments = len(boundaries) + 1
    return {
        "n_segments": int(n_segments),
        "mean_run_len": float(len(preds) / n_segments),
    }


def neighbor_label_consistency(E: np.ndarray, labels: np.ndarray,
                               window: int = 5) -> float:
    """Fraction of frames whose embedding-nearest temporal neighbors share its
    label. Label-aware stability: are frames close in TIME also close in
    FEATURE space with the same phase? High = smooth, discriminative features.
    Low = features jump around even within a single phase.

    This is the discriminative half of conditional_stability -- it goes to zero
    for a constant encoder (all neighbors identical -> ties broken arbitrarily)
    only if labels vary, which is exactly the degeneracy we want penalized.
    """
    T = E.shape[0]
    if T < 2 * window + 1:
        return float("nan")
    En = _l2norm(E)
    agree = 0
    for t in range(T):
        lo, hi = max(0, t - window), min(T, t + window + 1)
        idx = [j for j in range(lo, hi) if j != t]
        sims = En[idx] @ En[t]
        nearest = idx[int(np.argmax(sims))]
        agree += int(labels[nearest] == labels[t])
    return float(agree / T)


def conditional_stability(E: np.ndarray, labels: np.ndarray) -> dict:
    """The headline temporal metric, guarded against the degeneracy trap.

    Combines low drift (stable) with high neighbor-label consistency
    (discriminative). A constant encoder scores high on stability but its
    consistency collapses whenever labels vary within the window, so the
    product stays low. Reported as both components plus their product, so the
    trade-off is legible rather than hidden in one number.
    """
    # Use variance-NORMALIZED drift: raw drift rewards low-variance encoders
    # (see normalized_drift docstring). drift_ratio in ~[0,1+]; lower is more
    # stable, so stability = 1 - min(drift_ratio, 1).
    dr = normalized_drift(E)["drift_ratio"]
    stability = 1.0 - min(dr, 1.0)
    consistency = neighbor_label_consistency(E, labels)
    score = (float(stability * consistency)
             if not np.isnan(consistency) else float("nan"))
    return {
        "stability": float(stability),
        "consistency": float(consistency) if not np.isnan(consistency) else None,
        "conditional_score": score,
    }
