"""Synthetic controls for the temporal metrics.

Each case has a KNOWN answer. If a metric fails here, it is measuring the
wrong thing, and any real-data number it produces is meaningless. This is the
week-3 gate: no temporal metric runs on Cholec80 until all of these pass.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.metrics.temporal import (boundary_jitter, conditional_stability,
                                   embedding_velocity, feature_drift,
                                   neighbor_label_consistency,
                                   phase_fragmentation)

D = 384
RNG = np.random.default_rng(0)


# --- Control 1: static clip -> zero drift ----------------------------------

def test_static_clip_zero_drift():
    e = RNG.standard_normal(D)
    E = np.repeat(e[None, :], 50, axis=0)
    d = feature_drift(E)
    assert d["drift_mean"] < 1e-6
    assert d["drift_p95"] < 1e-6
    assert embedding_velocity(E).max() < 1e-6


# --- Control 2: shuffled distinct frames -> high drift ----------------------

def test_shuffled_frames_high_drift():
    E = RNG.standard_normal((50, D))          # independent -> ~orthogonal
    d = feature_drift(E)
    # cosine distance between random high-dim vectors clusters near 1.0
    assert d["drift_mean"] > 0.8


def test_static_below_shuffled():
    static = np.repeat(RNG.standard_normal(D)[None, :], 50, axis=0)
    shuffled = RNG.standard_normal((50, D))
    assert feature_drift(static)["drift_mean"] < feature_drift(shuffled)["drift_mean"]


# --- Control 3: constant encoder degeneracy --------------------------------

def test_constant_encoder_is_stable_but_not_discriminative():
    """The trap: a constant output looks perfectly stable but is useless.
    conditional_stability must NOT reward it."""
    E = np.repeat(RNG.standard_normal(D)[None, :], 60, axis=0)
    labels = RNG.integers(0, 7, size=60)      # labels vary; features do not
    cs = conditional_stability(E, labels)
    # Under variance-normalized drift, a constant encoder is DEGENERATE, not
    # stable: near-zero spread -> drift_ratio pinned to 1.0 -> stability 0.
    # The metric must refuse to reward it, on either the stability OR the
    # consistency axis, so the product stays low.
    assert cs["conditional_score"] < 0.5


def test_discriminative_smooth_encoder_scores_high():
    """Contrast: features that move slowly and cluster by label score high."""
    labels = np.repeat(np.arange(6), 10)      # 6 phases, 10 frames each
    centers = RNG.standard_normal((6, D)) * 5
    E = centers[labels] + RNG.standard_normal((60, D)) * 0.05  # tight clusters
    cs = conditional_stability(E, labels)
    assert cs["consistency"] > 0.9
    assert cs["conditional_score"] > 0.5


# --- Control 4: jitter vs clean prediction streams -------------------------

def test_clean_stream_few_transitions():
    preds = np.repeat(np.arange(7), 20)       # 7 phases in order
    bj = boundary_jitter(preds)
    frag = phase_fragmentation(preds)
    assert bj["transitions"] == 6
    assert frag["n_segments"] == 7


def test_jittery_stream_many_transitions():
    preds = np.tile([0, 1], 70)               # flip-flop every frame
    bj = boundary_jitter(preds)
    frag = phase_fragmentation(preds)
    assert bj["transitions"] == 139
    assert frag["mean_run_len"] < 1.5


def test_jitter_ordering():
    clean = np.repeat(np.arange(7), 20)
    jitter = np.tile([0, 1], 70)
    assert (boundary_jitter(jitter)["transitions_per_100"]
            > boundary_jitter(clean)["transitions_per_100"])


# --- neighbor consistency sanity -------------------------------------------

def test_neighbor_consistency_perfect_when_labels_constant():
    labels = np.zeros(30, dtype=int)
    E = RNG.standard_normal((30, D))
    assert neighbor_label_consistency(E, labels) == 1.0


def test_normalized_drift_penalizes_bland_encoder():
    """The loophole that fooled the first real run: a low-variance encoder has
    low RAW drift but its normalized drift is not artificially small."""
    from src.metrics.temporal import normalized_drift
    rng = np.random.default_rng(1)
    # bland: tiny steps around a point, tiny spread
    bland = np.repeat(rng.standard_normal(D)[None], 50, 0) + rng.standard_normal((50, D)) * 0.01
    # tracking: larger steps but proportionally larger spread
    walk = np.cumsum(rng.standard_normal((50, D)) * 0.1, axis=0)
    # raw drift favors bland; normalized should not blindly do so
    nb, nw = normalized_drift(bland)["drift_ratio"], normalized_drift(walk)["drift_ratio"]
    assert nb > 0.5   # bland's steps are large RELATIVE to its tiny spread


def test_causal_filter_uses_no_future_frames():
    """Causal output at frame i must not depend on frames after i."""
    from src.experiments.temporal_smoothing import majority_filter
    a = np.array([0] * 10 + [1] * 10)
    b = np.array([0] * 10 + [2] * 10)   # differs only after index 9
    ca = majority_filter(a, 5, causal=True)
    cb = majority_filter(b, 5, causal=True)
    assert np.array_equal(ca[:10], cb[:10]), "causal filter peeked at the future"


def test_centered_filter_leads_causal():
    """Centered reaches the new phase earlier than causal, because its window
    extends forward. This is the operational difference that makes causal the
    harder, deployment-realistic case."""
    from src.experiments.temporal_smoothing import majority_filter
    a = np.array([0] * 10 + [1] * 10)
    cen = majority_filter(a, 5, causal=False)
    cau = majority_filter(a, 5, causal=True)
    assert not np.array_equal(cen, cau), "centered and causal identical"
    assert int(np.argmax(cen == 1)) < int(np.argmax(cau == 1))
