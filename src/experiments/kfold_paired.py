"""5-fold cross-validated paired bootstrap over all 80 Cholec80 videos.

The 40/8/32 split gives only 32 paired differences, too few to resolve 1-3
point encoder margins. K-fold puts every video in a test set exactly once,
yielding 80 paired differences -- 2.5x the data for the paired test.

Critical: fold assignment is computed ONCE and shared across encoders. If
encoders saw different folds the differences would not be paired and the test
would be meaningless.

Trains n_folds probes per encoder (5 x 4 = 20 probes). Roughly an hour.

    python -m src.experiments.kfold_paired \
        --frames $BIGDIR/endoscopy/raw/cholec80_frames \
        --caches dinov2_vits14=<p> clip_vitb16=<p> \
                 endovit_vitb16=<p> biomedclip_vitb16=<p> \
        --causal
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

from src.experiments.paired_bootstrap import paired_boot
from src.experiments.temporal_smoothing import majority_filter


def _load(cdir, frames_dir, vid):
    E = np.asarray(np.load(Path(cdir) / f"{vid}.npy", mmap_mode="r"))
    lab = pd.read_csv(Path(frames_dir) / vid / "labels.csv")["phase_idx"].to_numpy()
    return E, lab


def make_folds(videos: list[str], k: int, seed: int = 0) -> dict[str, int]:
    """Shared fold assignment. Seeded and returned so it can be recorded."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(videos))
    return {videos[v]: int(i % k) for i, v in enumerate(perm)}


def score_encoder(cdir, frames_dir, folds, k, windows, causal):
    """-> {window: {video: (acc, macro_f1)}} with each video scored by the
    probe trained on the other k-1 folds."""
    out = {w: {} for w in windows}
    videos = sorted(folds)
    for f in range(k):
        train = [v for v in videos if folds[v] != f]
        test = [v for v in videos if folds[v] == f]
        X, y = [], []
        for v in train:
            e, l = _load(cdir, frames_dir, v)
            X.append(e); y.append(l)
        X, y = np.concatenate(X), np.concatenate(y)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 random_state=0).fit(sc.transform(X), y)
        for v in test:
            E, lab = _load(cdir, frames_dir, v)
            raw = clf.predict(sc.transform(E))
            for w in windows:
                p = majority_filter(raw, w, causal=causal)
                out[w][v] = (float((p == lab).mean()),
                             float(f1_score(lab, p, average="macro",
                                            zero_division=0)))
        print(f"    fold {f+1}/{k}: {len(train)} train / {len(test)} test videos")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 15, 31, 61, 121])
    ap.add_argument("--causal", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    caches = dict(kv.split("=", 1) for kv in args.caches)
    frames_dir = Path(args.frames)
    videos = sorted(p.stem for p in Path(next(iter(caches.values()))).glob("*.npy"))
    folds = make_folds(videos, args.k, args.seed)
    print(f"{len(videos)} videos, {args.k} folds, seed {args.seed}")
    print(f"fold sizes: {pd.Series(list(folds.values())).value_counts().sort_index().tolist()}\n")

    scores = {}
    for name, cdir in caches.items():
        print(f"{name}:")
        scores[name] = score_encoder(cdir, frames_dir, folds, args.k,
                                     args.windows, args.causal)

    mode = "causal" if args.causal else "centered"
    print("\n" + "=" * 78)
    print(f"K-FOLD PAIRED BOOTSTRAP  |  {mode}  |  {len(videos)} videos, "
          f"{args.k}-fold CV")
    print("=" * 78)

    # marginal accuracy per encoder, for the ranking table
    print("\nmean accuracy across all 80 videos:")
    hdr = "  ".join(f"w={w}" for w in args.windows)
    print(f"{'encoder':20} {hdr}")
    for name in caches:
        cells = "  ".join(
            f"{np.mean([scores[name][w][v][0] for v in videos]):.3f}"
            for w in args.windows)
        print(f"{name:20} {cells}")

    results = []
    for A, B in itertools.combinations(caches, 2):
        print(f"\n{A}  minus  {B}")
        print(f"{'w':>5} {'d_acc':>9} {'95% CI':>21} {'p':>8} {'A wins':>10}")
        for w in args.windows:
            d = np.array([scores[A][w][v][0] - scores[B][w][v][0] for v in videos])
            r = paired_boot(d)
            sig = "*" if r["p"] < 0.05 else " "
            print(f"{w:>5} {r['mean_diff']:>+9.4f} "
                  f"[{r['ci_low']:>+7.4f},{r['ci_high']:>+7.4f}] "
                  f"{r['p']:>8.4f}{sig} {r['n_videos_A_wins']:>4}/{len(videos)}")
            results.append({"pair": f"{A}:{B}", "window": w, "mode": mode,
                            "metric": "accuracy", **r})
            d2 = np.array([scores[A][w][v][1] - scores[B][w][v][1] for v in videos])
            results.append({"pair": f"{A}:{B}", "window": w, "mode": mode,
                            "metric": "macro_f1", **paired_boot(d2)})

    print("\n  * = 95% CI excludes zero")

    out = args.out or f"reports/kfold_paired_{mode}.json"
    Path(out).write_text(json.dumps(
        {"k": args.k, "seed": args.seed, "n_videos": len(videos),
         "folds": folds, "results": results,
         "per_video": {n: {str(w): {v: scores[n][w][v] for v in videos}
                           for w in args.windows} for n in caches}},
        indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
