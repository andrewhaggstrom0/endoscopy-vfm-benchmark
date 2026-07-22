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
    assert cs["stability"] > 0.99             # trivially stable
    # nearest neighbor is arbitrary among identical vectors -> consistency
    # tracks label base-rate, far below a discriminative model's ~1.0
    assert cs["consistency"] < 0.5
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
