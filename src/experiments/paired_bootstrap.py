"""Paired video-level bootstrap on smoothed-accuracy differences.

The headline rank inversion turns on small margins (DINOv2 0.782 vs EndoViT
0.777 at w=31 causal). This tests whether those margins survive video-level
resampling.

PAIRED, not two independent intervals: for each test video we compute
acc_A(video) - acc_B(video), then bootstrap that per-video difference.
Cholec80 videos vary enormously in length and difficulty, and both encoders
face the same videos, so pairing cancels that shared variation and is far more
powerful than comparing overlapping marginal CIs. Two overlapping CIs do NOT
imply a non-significant difference under pairing.

    python -m src.experiments.paired_bootstrap \
        --frames $BIGDIR/endoscopy/raw/cholec80_frames \
        --caches dinov2_vits14=<p> endovit_vitb16=<p> ... \
        --pairs dinov2_vits14:endovit_vitb16 \
        --windows 1 31 121 --causal
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.data.splits import ordered_split
from src.experiments.temporal_smoothing import majority_filter


def _load(cdir, frames_dir, vid):
    E = np.asarray(np.load(Path(cdir) / f"{vid}.npy", mmap_mode="r"))
    lab = pd.read_csv(Path(frames_dir) / vid / "labels.csv")["phase_idx"].to_numpy()
    return E, lab


def per_video_scores(cdir, frames_dir, train_ids, test_ids, windows, causal):
    """-> {window: {video: (acc, macro_f1)}}"""
    X, y = [], []
    for v in train_ids:
        e, l = _load(cdir, frames_dir, v)
        X.append(e); y.append(l)
    X, y = np.concatenate(X), np.concatenate(y)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000,
                             class_weight="balanced", random_state=0
                             ).fit(sc.transform(X), y)
    out = {w: {} for w in windows}
    for v in test_ids:
        E, lab = _load(cdir, frames_dir, v)
        raw = clf.predict(sc.transform(E))
        for w in windows:
            p = majority_filter(raw, w, causal=causal)
            out[w][v] = (float((p == lab).mean()),
                         float(f1_score(lab, p, average="macro", zero_division=0)))
    return out


def paired_boot(diffs: np.ndarray, n: int = 10000, seed: int = 0) -> dict:
    """Bootstrap the mean of per-video paired differences."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n, len(diffs)))
    means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    # two-sided bootstrap p: fraction of resamples on the wrong side of zero
    p = 2 * min((means <= 0).mean(), (means >= 0).mean())
    return {"mean_diff": float(diffs.mean()),
            "ci_low": float(lo), "ci_high": float(hi),
            "p": float(min(p, 1.0)),
            "n_videos": int(len(diffs)),
            "n_videos_A_wins": int((diffs > 0).sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--pairs", nargs="+", required=True,
                    help="A:B pairs; tests acc(A) - acc(B)")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 15, 31, 61, 121])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--protocol", default="train40_val8_test32")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    caches = dict(kv.split("=", 1) for kv in args.caches)
    frames_dir = Path(args.frames)
    split = ordered_split(
        sorted(p.stem for p in Path(next(iter(caches.values()))).glob("*.npy")),
        args.protocol)
    train_ids, test_ids = split.train + split.val, split.test

    scores = {}
    for name, cdir in caches.items():
        print(f"scoring {name} ...")
        scores[name] = per_video_scores(cdir, frames_dir, train_ids, test_ids,
                                        args.windows, args.causal)

    mode = "causal" if args.causal else "centered"
    print("\n" + "=" * 76)
    print(f"PAIRED VIDEO-LEVEL BOOTSTRAP  |  {mode} smoothing  |  "
          f"{len(test_ids)} test videos")
    print("=" * 76)

    results = []
    for pair in args.pairs:
        A, B = pair.split(":")
        print(f"\n{A}  minus  {B}")
        print(f"{'w':>5} {'d_acc':>8} {'95% CI':>20} {'p':>8} {'A wins':>9}")
        for w in args.windows:
            d = np.array([scores[A][w][v][0] - scores[B][w][v][0] for v in test_ids])
            r = paired_boot(d)
            sig = "*" if r["p"] < 0.05 else " "
            print(f"{w:>5} {r['mean_diff']:>+8.4f} "
                  f"[{r['ci_low']:>+7.4f},{r['ci_high']:>+7.4f}] "
                  f"{r['p']:>8.4f}{sig} {r['n_videos_A_wins']:>4}/{len(test_ids)}")
            results.append({"pair": pair, "window": w, "mode": mode,
                            "metric": "accuracy", **r})
            # macro-F1 too
            d2 = np.array([scores[A][w][v][1] - scores[B][w][v][1] for v in test_ids])
            results.append({"pair": pair, "window": w, "mode": mode,
                            "metric": "macro_f1", **paired_boot(d2)})

    print("\n  * = 95% CI excludes zero")
    print("  Note: a consistent SIGN across windows is evidence even where")
    print("  individual windows are not separately significant.")

    out = args.out or f"reports/paired_bootstrap_{mode}.json"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
