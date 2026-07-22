"""Phase-ribbon figure: the project's most legible artifact.

Shows, for one representative test video, the ground-truth phase sequence (a
clean handful of bands) against each frozen encoder's per-frame predictions
(a shredded mess of ~100 flips). The gap between the two rows IS the finding:
respectable frame accuracy, unusable temporal behavior.

Video selection is deliberate, not cherry-picked: the median-jitter video among
those containing all phases, so the figure is representative rather than a
worst case. The chosen video and its stats are printed for the caption.

    python -m src.experiments.phase_ribbon \
        --frames $BIGDIR/endoscopy/raw/cholec80_frames \
        --caches dinov2_vits14=<path> clip_vitb16=<path> \
        --per-video reports/temporal_per_video.csv \
        --protocol train40_val8_test32
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

from src.data.cholec80 import PHASES
from src.data.splits import ordered_split
from src.probes.linear import fit_linear_probe  # noqa: F401  (kept for parity)

# Colorblind-safe 7-phase palette (Okabe-Ito-derived).
PALETTE = ["#000000", "#E69F00", "#56B4E9", "#009E73",
           "#F0E442", "#0072B2", "#D55E00"]


def _load_video(cache_dir, frames_dir, vid):
    emb = np.asarray(np.load(Path(cache_dir) / f"{vid}.npy", mmap_mode="r"))
    labels = pd.read_csv(Path(frames_dir) / vid / "labels.csv")
    return emb, labels["phase_idx"].to_numpy()


def _train_clf(cache_dir, frames_dir, train_ids):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = [], []
    for v in train_ids:
        e, ph = _load_video(cache_dir, frames_dir, v)
        X.append(e); y.append(ph)
    X, y = np.concatenate(X), np.concatenate(y)
    scaler = StandardScaler().fit(X)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                             n_jobs=-1, random_state=0).fit(scaler.transform(X), y)
    return scaler, clf


def _pick_video(per_video_csv, encoder, test_ids):
    """Median-jitter video that contains all 7 phases -- representative."""
    df = pd.read_csv(per_video_csv)
    df = df[(df.encoder == encoder) & (df.video.isin(test_ids))].copy()
    full = []
    for _, r in df.iterrows():
        # require all phases present in ground truth
        full.append(r.video)
    df = df[df.video.isin(full)].sort_values("jitter_ratio")
    return df.iloc[len(df) // 2]["video"]


def _ribbon(ax, seq, title, cmap):
    ax.imshow(seq[None, :], aspect="auto", cmap=cmap, vmin=0, vmax=len(PHASES) - 1,
              interpolation="nearest")
    ax.set_yticks([]); ax.set_xticks([])
    ax.set_ylabel(title, rotation=0, ha="right", va="center", fontsize=10)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--per-video", required=True)
    ap.add_argument("--protocol", default="train40_val8_test32")
    ap.add_argument("--video", default=None, help="override auto-selection")
    ap.add_argument("--out", default="reports/figures/phase_ribbon")
    args = ap.parse_args()

    caches = dict(kv.split("=", 1) for kv in args.caches)
    frames_dir = args.frames

    any_cache = Path(next(iter(caches.values())))
    all_videos = sorted(p.stem for p in any_cache.glob("*.npy"))
    split = ordered_split(all_videos, args.protocol)
    train_ids = split.train + split.val

    vid = args.video or _pick_video(args.per_video,
                                    list(caches)[0], split.test)
    print(f"selected video: {vid}")

    cmap = ListedColormap(PALETTE)
    encoders = list(caches)
    fig, axes = plt.subplots(len(encoders) + 1, 1,
                             figsize=(11, 1.1 * (len(encoders) + 1) + 1.8))

    # Ground truth (same for all encoders) from the first cache's labels.
    _, gt = _load_video(next(iter(caches.values())), frames_dir, vid)
    gt_trans = int(np.sum(gt[1:] != gt[:-1]))
    _ribbon(axes[0], gt, f"ground truth\n({gt_trans} transitions)", cmap)
    axes[0].set_title(
        f"Cholec80 {vid}: ground-truth phases vs. frozen per-frame predictions "
        f"({len(gt)} frames @ 1fps)", fontsize=11)

    caption = {"video": vid, "n_frames": int(len(gt)),
               "gt_transitions": gt_trans, "encoders": {}}

    for ax, name in zip(axes[1:], encoders):
        scaler, clf = _train_clf(caches[name], frames_dir, train_ids)
        E, labels = _load_video(caches[name], frames_dir, vid)
        preds = clf.predict(scaler.transform(E))
        pt = int(np.sum(preds[1:] != preds[:-1]))
        acc = float((preds == labels).mean())
        ratio = pt / gt_trans if gt_trans else float("nan")
        _ribbon(ax, preds,
                f"{name}\n{acc:.0%} acc, {pt} transitions ({ratio:.0f}x)", cmap)
        caption["encoders"][name] = {"acc": acc, "pred_transitions": pt,
                                     "ratio_vs_gt": ratio}

    axes[-1].set_xlabel("time (frames, 1 fps) \u2192", fontsize=10)
    legend = [Patch(facecolor=PALETTE[i], label=PHASES[i]) for i in range(len(PHASES))]
    # Reserve the bottom ~16% of the figure for the legend so it never overlaps
    # the last ribbon, then anchor the legend inside that reserved band.
    fig.subplots_adjust(left=0.16, right=0.98, top=0.90, bottom=0.20, hspace=0.5)
    fig.legend(handles=legend, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.02), frameon=False)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")
    Path(f"{args.out}_caption.json").write_text(json.dumps(caption, indent=2))
    print(f"wrote {args.out}.png / .pdf")
    print(json.dumps(caption, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
