"""Figure 2: the rank inversion under temporal smoothing.

Two panels -- centered (offline) and causal (real-time) -- showing frame
accuracy against smoothing window, one line per encoder. The crossover where
the frame-level winner is overtaken is detected programmatically and annotated,
so the figure stays correct if encoders are added.

Jitter is drawn on a secondary axis (log scale, dashed) to show the two things
moving together: as jitter collapses, the ranking flips.

    python -m src.experiments.figure_smoothing \
        --centered reports/temporal_smoothing.json \
        --causal reports/temporal_smoothing_causal.json
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

COLORS = {"endovit_vitb16": "#D55E00",
          "dinov2_vits14": "#0072B2",
          "clip_vitb16": "#009E73"}
LABELS = {"endovit_vitb16": "EndoViT (surgical)",
          "dinov2_vits14": "DINOv2 (self-sup.)",
          "clip_vitb16": "CLIP (image-text)"}


def find_crossover(df: pd.DataFrame) -> tuple | None:
    """First window where the w=1 leader is no longer top by accuracy."""
    windows = sorted(df.window.unique())
    first = df[df.window == windows[0]]
    leader = first.loc[first.accuracy.idxmax(), "encoder"]
    for w in windows[1:]:
        d = df[df.window == w]
        top = d.loc[d.accuracy.idxmax(), "encoder"]
        if top != leader:
            return w, leader, top
    return None


def panel(ax, df, title, annotate=True):
    windows = sorted(df.window.unique())
    for enc in df.encoder.unique():
        d = df[df.encoder == enc].sort_values("window")
        ax.plot(d.window, d.accuracy, "-o", ms=4, lw=2,
                color=COLORS.get(enc, None), label=LABELS.get(enc, enc))
    ax.set_xscale("log")
    ax.set_xticks(windows)
    ax.set_xticklabels([str(w) for w in windows])
    ax.set_xlabel("smoothing window (frames @ 1 fps)")
    ax.set_ylabel("frame accuracy")
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.25, which="both")

    # jitter on a twin axis, dashed, log scale
    ax2 = ax.twinx()
    for enc in df.encoder.unique():
        d = df[df.encoder == enc].sort_values("window")
        ax2.plot(d.window, d.jitter_ratio, "--", lw=1, alpha=0.45,
                 color=COLORS.get(enc, None))
    ax2.set_yscale("log")
    ax2.set_ylabel("jitter ratio (x ground truth, dashed)", fontsize=9)
    ax2.tick_params(labelsize=8)

    cx = find_crossover(df)
    if cx and annotate:
        w, old, new = cx
        ax.axvline(w, color="0.35", ls=":", lw=1.2)
        ymin, ymax = ax.get_ylim()
        ax.annotate(f"ranking flips\n{LABELS[old].split()[0]} \u2192 "
                    f"{LABELS[new].split()[0]}",
                    xy=(w, ymin + 0.08 * (ymax - ymin)),
                    xytext=(w * 1.5, ymin + 0.02 * (ymax - ymin)),
                    fontsize=8, color="0.25",
                    arrowprops=dict(arrowstyle="->", color="0.45", lw=1))
    return cx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--centered", default="reports/temporal_smoothing.json")
    ap.add_argument("--causal", default="reports/temporal_smoothing_causal.json")
    ap.add_argument("--out", default="reports/figures/smoothing_inversion")
    args = ap.parse_args()

    cen = pd.DataFrame(json.loads(Path(args.centered).read_text()))
    cau = pd.DataFrame(json.loads(Path(args.causal).read_text()))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    c1 = panel(axes[0], cen, "Centered window (offline / retrospective)")
    c2 = panel(axes[1], cau, "Trailing window (real-time / deployment)")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.subplots_adjust(left=0.07, right=0.93, top=0.86, bottom=0.24, wspace=0.42)
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, 0.02))
    fig.suptitle("Frame-level accuracy selects a different backbone than "
                 "temporally-smoothed accuracy", fontsize=12, y=0.97)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=200, bbox_inches="tight")

    caption = {"centered_crossover": c1, "causal_crossover": c2}
    Path(f"{args.out}_caption.json").write_text(json.dumps(caption, indent=2,
                                                           default=str))
    print(f"wrote {args.out}.png / .pdf")
    print("centered crossover:", c1)
    print("causal   crossover:", c2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
