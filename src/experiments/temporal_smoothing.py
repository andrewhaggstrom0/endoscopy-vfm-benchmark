"""How much jitter does trivial temporal smoothing recover?

Sweeps a centered majority filter over the per-frame prediction stream and
reports accuracy, macro-F1, and jitter ratio as a function of window size.

The expected shape -- accuracy roughly flat, jitter collapsing -- is the
thesis quantified: the phase information was in the frozen features all along,
and frame-level accuracy gave no warning of how much temporal repair the
predictions would need. Two encoders at identical accuracy can differ by an
order of magnitude in required smoothing, and the standard metric is blind
to it.

    python -m src.experiments.temporal_smoothing \
        --frames $BIGDIR/endoscopy/raw/cholec80_frames \
        --caches dinov2_vits14=<p> clip_vitb16=<p> endovit_vitb16=<p>
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


def majority_filter(preds: np.ndarray, w: int, causal: bool = False) -> np.ndarray:
    """Majority vote over a window of w frames (w=1 = identity).

    causal=False: centered window [i-w/2, i+w/2]. Uses FUTURE frames, so it is
      an offline smoother -- the generous case.
    causal=True: trailing window [i-w+1, i]. Only past frames, which is the
      only option for a real-time surgical system. Strictly harder.

    The headline rank inversion must be verified under causal smoothing; if it
    only holds for the centered filter, it is an artifact of hindsight.
    """
    if w <= 1:
        return preds.copy()
    out = np.empty_like(preds)
    for i in range(len(preds)):
        if causal:
            lo, hi = max(0, i - w + 1), i + 1
        else:
            half = w // 2
            lo, hi = max(0, i - half), min(len(preds), i + half + 1)
        out[i] = Counter(preds[lo:hi]).most_common(1)[0][0]
    return out


def _load(cdir, frames_dir, vid):
    E = np.asarray(np.load(Path(cdir) / f"{vid}.npy", mmap_mode="r"))
    lab = pd.read_csv(Path(frames_dir) / vid / "labels.csv")["phase_idx"].to_numpy()
    return E, lab


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--protocol", default="train40_val8_test32")
    ap.add_argument("--windows", type=int, nargs="+",
                    default=[1, 3, 5, 9, 15, 31, 61, 121])
    ap.add_argument("--causal", action="store_true",
                    help="trailing window only (real-time deployment case)")
    ap.add_argument("--out", default="reports/temporal_smoothing.json")
    args = ap.parse_args()

    caches = dict(kv.split("=", 1) for kv in args.caches)
    frames_dir = Path(args.frames)
    any_cache = Path(next(iter(caches.values())))
    split = ordered_split(sorted(p.stem for p in any_cache.glob("*.npy")),
                          args.protocol)
    train_ids, test_ids = split.train + split.val, split.test

    rows = []
    for name, cdir in caches.items():
        X, y = [], []
        for v in train_ids:
            e, l = _load(cdir, frames_dir, v)
            X.append(e); y.append(l)
        X, y = np.concatenate(X), np.concatenate(y)
        sc = StandardScaler().fit(X)
        clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                                 random_state=0).fit(sc.transform(X), y)
        print(f"{name}: probe fitted")

        for w in args.windows:
            accs, f1s, ratios = [], [], []
            for v in test_ids:
                E, lab = _load(cdir, frames_dir, v)
                p = majority_filter(clf.predict(sc.transform(E)), w,
                                    causal=args.causal)
                gt_t = int(np.sum(lab[1:] != lab[:-1]))
                pr_t = int(np.sum(p[1:] != p[:-1]))
                accs.append(float((p == lab).mean()))
                f1s.append(f1_score(lab, p, average="macro", zero_division=0))
                ratios.append(pr_t / gt_t if gt_t else np.nan)
            rows.append({"encoder": name, "window": w,
                         "accuracy": float(np.mean(accs)),
                         "macro_f1": float(np.mean(f1s)),
                         "jitter_ratio": float(np.nanmean(ratios))})
            print(f"  w={w:4}  acc={rows[-1]['accuracy']:.3f}  "
                  f"F1={rows[-1]['macro_f1']:.3f}  "
                  f"jitter={rows[-1]['jitter_ratio']:.1f}x")

    df = pd.DataFrame(rows)
    df.to_csv("reports/temporal_smoothing.csv", index=False)

    print("\n" + "=" * 64)
    mode = "CAUSAL (trailing)" if args.causal else "CENTERED (offline)"
    print(f"TEMPORAL SMOOTHING SWEEP  |  {mode} majority filter")
    print("=" * 64)
    for enc in df.encoder.unique():
        d = df[df.encoder == enc]
        base, best = d.iloc[0], d.loc[d.jitter_ratio.idxmin()]
        print(f"\n{enc}")
        print(f"  w=1   : acc {base.accuracy:.3f}  jitter {base.jitter_ratio:>6.1f}x")
        print(f"  w={int(best.window):<4}: acc {best.accuracy:.3f}  "
              f"jitter {best.jitter_ratio:>6.1f}x")
        print(f"  -> jitter {base.jitter_ratio/max(best.jitter_ratio,1e-9):.0f}x lower "
              f"for {(best.accuracy-base.accuracy)*100:+.1f} pt accuracy")
        # smallest window reaching a deployable-ish 2x over-segmentation
        ok = d[d.jitter_ratio <= 2.0]
        if len(ok):
            r = ok.iloc[0]
            print(f"  window to reach <=2x jitter: {int(r.window)} frames "
                  f"({int(r.window)}s at 1fps), acc {r.accuracy:.3f}")
        else:
            print("  never reaches <=2x jitter within the swept windows")

    Path(args.out).write_text(json.dumps(df.to_dict(orient="records"), indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
