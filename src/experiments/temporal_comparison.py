"""Temporal comparison across encoders -- the project's core experiment.

For each encoder, for each Cholec80 test video, measures temporal stability
WITHIN that video (never across videos), then aggregates at the video level.
Also runs the frozen probe per video to get a prediction stream, so boundary
jitter can be compared against each video's TRUE phase-transition count.

The headline output: do encoders rank the same by probe accuracy as by
temporal stability? A divergence (rank inversion) is the finding the whole
project is built to surface.

    python -m src.experiments.temporal_comparison \
        --frames $BIGDIR/endoscopy/raw/cholec80_frames \
        --caches dinov2_vits14=$BIGDIR/.../dinov2_vits14/<hash>/cholec80 \
                 clip_vitb16=$BIGDIR/.../clip_vitb16/<hash>/cholec80 \
        --protocol train40_val8_test32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.data.cholec80 import PHASES
from src.data.splits import assert_no_leakage, ordered_split
from src.metrics.temporal import (boundary_jitter, conditional_stability,
                                   feature_drift, phase_fragmentation)
from src.probes.linear import fit_linear_probe


def _load_video(cache_dir, frames_dir, vid):
    meta = json.loads((cache_dir / f"{vid}.json").read_text())
    emb = np.asarray(np.load(cache_dir / f"{vid}.npy", mmap_mode="r"))
    labels = pd.read_csv(frames_dir / vid / "labels.csv")
    if len(emb) != len(labels):
        raise ValueError(f"{vid}: {len(emb)} emb vs {len(labels)} labels")
    return emb, labels["phase_idx"].to_numpy()


def _assemble(cache_dir, frames_dir, vids):
    X, y, vof = [], [], []
    for v in vids:
        e, ph = _load_video(cache_dir, frames_dir, v)
        X.append(e); y.append(ph); vof.extend([v] * len(e))
    return np.concatenate(X), np.concatenate(y), vof


def _true_transition_rate(labels: np.ndarray) -> float:
    return 100.0 * np.sum(labels[1:] != labels[:-1]) / (len(labels) - 1)


def run_encoder(name, cache_dir, frames_dir, split):
    cache_dir = Path(cache_dir)
    train_ids = split.train + split.val
    test_ids = split.test

    # One probe per encoder, trained on train, applied per test video so each
    # video gets its own prediction stream (needed for boundary jitter).
    Xtr, ytr, _ = _assemble(cache_dir, frames_dir, train_ids)
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             random_state=0).fit(scaler.transform(Xtr), ytr)

    rows = []
    for vid in test_ids:
        E, labels = _load_video(cache_dir, frames_dir, vid)
        preds = clf.predict(scaler.transform(E))

        drift = feature_drift(E)                      # within-video only
        jit = boundary_jitter(preds)
        frag = phase_fragmentation(preds)
        cs = conditional_stability(E, labels)
        true_rate = _true_transition_rate(labels)

        rows.append({
            "encoder": name, "video": vid, "n_frames": len(E),
            "frame_acc": float((preds == labels).mean()),
            "drift_mean": drift["drift_mean"],
            "drift_p95": drift["drift_p95"],
            "pred_transitions_per_100": jit["transitions_per_100"],
            "true_transitions_per_100": true_rate,
            "jitter_ratio": (jit["transitions_per_100"] / true_rate
                             if true_rate > 0 else np.nan),
            "pred_segments": frag["n_segments"],
            "mean_run_len": frag["mean_run_len"],
            "stability": cs["stability"],
            "consistency": cs["consistency"],
            "conditional_score": cs["conditional_score"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--caches", nargs="+", required=True,
                    help="name=path pairs")
    ap.add_argument("--protocol", default="train40_val8_test32")
    ap.add_argument("--out", default="reports/temporal_comparison.json")
    ap.add_argument("--csv", default="reports/temporal_per_video.csv")
    args = ap.parse_args()

    frames_dir = Path(args.frames)
    caches = dict(kv.split("=", 1) for kv in args.caches)

    any_cache = Path(next(iter(caches.values())))
    all_videos = sorted(p.stem for p in any_cache.glob("*.npy"))
    split = ordered_split(all_videos, args.protocol)
    assert_no_leakage(split)
    print(f"{split.summary()}\ntest videos: {len(split.test)}\n")

    per_video = pd.concat(
        [run_encoder(n, c, frames_dir, split) for n, c in caches.items()],
        ignore_index=True)
    per_video.to_csv(args.csv, index=False)

    # Video-level aggregates per encoder.
    agg = per_video.groupby("encoder").agg(
        frame_acc=("frame_acc", "mean"),
        drift_mean=("drift_mean", "mean"),
        jitter_ratio=("jitter_ratio", "mean"),
        conditional_score=("conditional_score", "mean"),
        mean_run_len=("mean_run_len", "mean"),
    ).reset_index()

    print("=" * 70)
    print("TEMPORAL COMPARISON  |  Cholec80 test  |  per-video, within-sequence")
    print("=" * 70)
    print(f"{'encoder':16} {'frame_acc':>10} {'drift':>8} "
          f"{'jitter_x':>9} {'cond_score':>11} {'run_len':>8}")
    for _, r in agg.iterrows():
        print(f"{r['encoder']:16} {r['frame_acc']:>10.3f} {r['drift_mean']:>8.4f} "
              f"{r['jitter_ratio']:>9.1f} {r['conditional_score']:>11.3f} "
              f"{r['mean_run_len']:>8.1f}")

    # The headline: does probe-accuracy ranking match stability ranking?
    ranked_by_acc = agg.sort_values("frame_acc", ascending=False)["encoder"].tolist()
    ranked_by_stab = agg.sort_values("conditional_score", ascending=False)["encoder"].tolist()
    ranked_by_jitter = agg.sort_values("jitter_ratio", ascending=True)["encoder"].tolist()

    print("\nrankings:")
    print(f"  by frame accuracy    : {ranked_by_acc}")
    print(f"  by conditional stab. : {ranked_by_stab}")
    print(f"  by jitter (low=good) : {ranked_by_jitter}")

    inverted = ranked_by_acc != ranked_by_stab
    print(f"\n  RANK INVERSION (acc vs stability): {inverted}")
    if inverted:
        print("  -> the finding: frame accuracy and temporal stability DISAGREE.")
    else:
        print("  -> rankings agree; frame accuracy tracks stability here.")

    # If >=3 encoders, a real correlation; with 2 it's just concordant/not.
    rho = p = None
    if len(agg) >= 3:
        rho, p = spearmanr(agg["frame_acc"], agg["conditional_score"])
        print(f"\n  Spearman(acc, stability) = {rho:.3f}  (p={p:.3f})")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "protocol": args.protocol,
        "aggregates": agg.to_dict(orient="records"),
        "ranked_by_accuracy": ranked_by_acc,
        "ranked_by_stability": ranked_by_stab,
        "ranked_by_jitter": ranked_by_jitter,
        "rank_inversion": bool(inverted),
        "spearman_rho": (float(rho) if rho is not None else None),
        "spearman_p": (float(p) if p is not None else None),
    }, indent=2))
    print(f"\nwrote {args.out} and {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
