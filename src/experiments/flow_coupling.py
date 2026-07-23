"""Per-encoder embedding-velocity vs optical-flow coupling on Cholec80.

    python -m src.experiments.flow_coupling \
        --frames $BIGDIR/endoscopy/raw/cholec80_frames \
        --caches dinov2_vits14=<p> clip_vitb16=<p> endovit_vitb16=<p> \
        --n-videos 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.splits import ordered_split
from src.metrics.flow import flow_magnitude_sequence, velocity_flow_coupling
from src.metrics.temporal import embedding_velocity


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", required=True)
    ap.add_argument("--caches", nargs="+", required=True)
    ap.add_argument("--protocol", default="train40_val8_test32")
    ap.add_argument("--n-videos", type=int, default=10,
                    help="test videos to sample (flow is the expensive pass)")
    ap.add_argument("--out", default="reports/flow_coupling.json")
    args = ap.parse_args()

    caches = dict(kv.split("=", 1) for kv in args.caches)
    frames_dir = Path(args.frames)
    any_cache = Path(next(iter(caches.values())))
    all_videos = sorted(p.stem for p in any_cache.glob("*.npy"))
    split = ordered_split(all_videos, args.protocol)
    # deterministic, evenly spaced sample of test videos
    test = split.test
    idx = np.linspace(0, len(test) - 1, min(args.n_videos, len(test))).astype(int)
    vids = [test[i] for i in idx]
    print(f"flow on {len(vids)} test videos: {vids}\n")

    # Flow is encoder-independent: compute once per video, reuse.
    flow_cache = {}
    for v in vids:
        paths = sorted((frames_dir / v).glob("frame_*.jpg"),
                       key=lambda p: int("".join(c for c in p.stem if c.isdigit())))
        flow_cache[v] = flow_magnitude_sequence(paths)
        print(f"  {v}: {len(flow_cache[v])} flow pairs, "
              f"mean mag {flow_cache[v].mean():.3f}")

    rows = []
    for name, cdir in caches.items():
        for v in vids:
            E = np.asarray(np.load(Path(cdir) / f"{v}.npy", mmap_mode="r"))
            vel = embedding_velocity(E)
            c = velocity_flow_coupling(vel, flow_cache[v])
            rows.append({"encoder": name, "video": v, **c,
                         "vel_mean": float(vel.mean()),
                         "flow_mean": float(flow_cache[v].mean())})

    df = pd.DataFrame(rows)
    agg = df.groupby("encoder").agg(
        rho_mean=("rho", "mean"), rho_std=("rho", "std"),
        frac_sig=("p", lambda s: float((s < 0.05).mean())),
    ).reset_index()

    print("\n" + "=" * 60)
    print("EMBEDDING VELOCITY  vs  OPTICAL FLOW  (Spearman, per video)")
    print("=" * 60)
    print(f"{'encoder':18} {'rho':>8} {'sd':>8} {'frac p<.05':>12}")
    for _, r in agg.iterrows():
        print(f"{r['encoder']:18} {r['rho_mean']:>8.3f} {r['rho_std']:>8.3f} "
              f"{r['frac_sig']:>12.2f}")
    print("\n  rho -> 1 : drift tracks scene motion (signal)")
    print("  rho -> 0 : drift is independent of the scene (noise)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "videos": vids,
        "per_video": df.to_dict(orient="records"),
        "aggregate": agg.to_dict(orient="records"),
    }, indent=2))
    df.to_csv("reports/flow_coupling_per_video.csv", index=False)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
