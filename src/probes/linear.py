"""Frozen-feature linear probe for phase recognition.

The standard frozen-backbone protocol: features are fixed, only a linear
classifier is trained. This measures how linearly separable the phases are in
the encoder's representation -- nothing more. A low number here is the premise
of this project, not a failure: it is why the field bolts temporal heads
(TeCNO, Trans-SVNet) on top of frozen features.

Reported metrics
----------------
- frame-level accuracy   (comparable to published frozen-feature baselines)
- macro-F1               (the honest number under Cholec80's ~31x phase
                          imbalance; raw accuracy is a vanity metric here)
- balanced accuracy
- per-phase F1           (shows *which* phases collapse -- usually the rare
                          Preparation / CleaningCoagulation)

Standardization is fit on train only. Fitting it on the full set leaks test
statistics into the probe and is a subtle, common way to inflate results.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             f1_score)
from sklearn.preprocessing import StandardScaler


@dataclass
class ProbeResult:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float
    per_phase_f1: dict[str, float]
    # kept for video-level bootstrap; not printed
    test_video_ids: list[str] = field(default_factory=list)
    test_pred: np.ndarray | None = None
    test_true: np.ndarray | None = None


def fit_linear_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    phase_names: list[str],
    test_video_of_frame: list[str],
    C: float = 1.0,
    max_iter: int = 2000,
    seed: int = 0,
) -> ProbeResult:
    scaler = StandardScaler().fit(X_train)          # train only
    Xtr = scaler.transform(X_train)
    Xte = scaler.transform(X_test)

    clf = LogisticRegression(
        C=C, max_iter=max_iter,
        # multinomial is the default in sklearn >=1.7; the explicit
        # multi_class arg was removed. Solver 'lbfgs' handles it.
        class_weight="balanced",   # counter the phase imbalance in the loss
        n_jobs=-1, random_state=seed,
    ).fit(Xtr, y_train)

    pred = clf.predict(Xte)
    present = sorted(set(y_test))
    per_phase = f1_score(y_test, pred, labels=present,
                         average=None, zero_division=0)
    return ProbeResult(
        accuracy=float(accuracy_score(y_test, pred)),
        macro_f1=float(f1_score(y_test, pred, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_test, pred)),
        per_phase_f1={phase_names[i]: float(f) for i, f in zip(present, per_phase)},
        test_video_ids=test_video_of_frame,
        test_pred=pred,
        test_true=y_test,
    )
