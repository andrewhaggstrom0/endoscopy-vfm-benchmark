"""Frozen-feature kNN probe -- label-free representation quality.

kNN needs no trained head, so it reflects the geometry of the embedding space
directly: if same-phase frames cluster, kNN succeeds. This is the number the
surgical-SSL literature reports (k=20, cosine) alongside linear probing, so it
is the more directly comparable of the two for a reproduction check.

Cosine metric because DINOv2 CLS features are compared by angle throughout this
project; using Euclidean here would measure a different geometry than the
temporal-drift metrics do.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import normalize


def fit_knn_probe(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    k: int = 20,
) -> dict:
    # L2-normalize so Euclidean kNN == cosine kNN; sklearn's cosine metric is
    # slow on 100k+ points, this is the standard equivalent trick.
    Xtr = normalize(X_train)
    Xte = normalize(X_test)
    clf = KNeighborsClassifier(n_neighbors=k, metric="euclidean",
                               n_jobs=-1).fit(Xtr, y_train)
    pred = clf.predict(Xte)
    return {
        "accuracy": float(accuracy_score(y_test, pred)),
        "macro_f1": float(f1_score(y_test, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, pred)),
        "k": k,
    }
