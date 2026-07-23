"""Week-2 reproduction gate.

Loads cached DINOv2 embeddings, assembles a video-level 40/8/32 split, runs the
linear and kNN probes on frame-level phase labels, and reports accuracy and
macro-F1 with VIDEO-LEVEL bootstrap CIs.

    python -m src.experiments.reproduction_gate \
        --cache $BIGDIR/endoscopy/cache/dinov2_vits14/<hash>/cholec80 \
        --frames $BIGDIR/endoscopy/raw/cholec80_frames \
        --protocol train40_val8_test32

The gate PASSES if:
  * assert_no_leakage holds (built in -- a crash here is the point), and
  * frame-level accuracy lands in the frozen-feature neighborhood
    (~55-75%), NOT near the 90%+ temporal-model results.

A number above ~85% frame-level is a leakage signal, not a triumph. A number
in the 60s is the expected, thesis-supporting outcome: frozen features are
temporally unlabelled and phase boundaries blur.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.cache.store import EmbeddingStore
from src.data.cholec80 import PHASES
from src.data.splits import assert_no_leakage, ordered_split, split_units
from src.probes.knn import fit_knn_probe
from src.probes.linear import fit_linear_probe


def _load_video(cache_dir: Path, frames_dir: Path, vid: str):
    """Return (embeddings, phase_idx) aligned frame-for-frame.

    Alignment is asserted, not assumed: the shard's frame_ids must line up with
    the labels.csv rows, or the whole result is silently meaningless.
    """
    meta = json.loads((cache_dir / f"{vid}.json").read_text())
    emb = np.load(cache_dir / f"{vid}.npy", mmap_mode="r")
    labels = pd.read_csv(frames_dir / vid / "labels.csv")

    if len(emb) != len(labels):
        raise ValueError(f"{vid}: {len(emb)} embeddings vs {len(labels)} labels")
    # frame_ids are frame_XXXXXX; labels are ordered by native_frame ascending.
    shard_frames = [int("".join(c for c in f if c.isdigit()))
                    for f in meta["frame_ids"]]
    if shard_frames != labels["native_frame"].tolist():
        raise ValueError(f"{vid}: frame_id / label order mismatch")
    return np.asarray(emb), labels["phase_idx"].to_numpy()


def _assemble(cache_dir, frames_dir, video_ids):
    X, y, vof = [], [], []
    for vid in video_ids:
        emb, ph = _load_video(cache_dir, frames_dir, vid)
        X.append(emb)
        y.append(ph)
        vof.extend([vid] * len(emb))
    return np.concatenate(X), np.concatenate(y), vof


def _video_bootstrap(video_ids, pred, true, vof, n=1000, seed=0):
    """Resample VIDEOS (not frames) and recompute macro-F1 + accuracy.

    Effective n is the number of surgeries, ~40 in test -- a frame-level
    bootstrap would report fake-tight intervals over 100k near-duplicates.
    """
    from sklearn.metrics import accuracy_score, f1_score
    vof = np.asarray(vof)
    by_video = {v: np.where(vof == v)[0] for v in video_ids}
    rng = np.random.default_rng(seed)
    accs, f1s = [], []
    for _ in range(n):
        pick = rng.choice(video_ids, size=len(video_ids), replace=True)
        idx = np.concatenate([by_video[v] for v in pick])
        accs.append(accuracy_score(true[idx], pred[idx]))
        f1s.append(f1_score(true[idx], pred[idx], average="macro", zero_division=0))
    ci = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"acc_ci": ci(accs), "f1_ci": ci(f1s)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--protocol", default="train40_val8_test32")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--out", default="reports/reproduction_gate.json")
    args = ap.parse_args()

    cache_dir, frames_dir = Path(args.cache), Path(args.frames)
    all_videos = sorted(p.stem for p in cache_dir.glob("*.npy"))
    print(f"videos in cache: {len(all_videos)}")

    split = ordered_split(all_videos, args.protocol)
    assert_no_leakage(split)                    # the gate's guardrail
    print(split.summary())

    # train = train ∪ val for the probe (val unused here; no HP tuning)
    train_ids = split.train + split.val
    test_ids = split.test

    print("assembling train ...")
    Xtr, ytr, _ = _assemble(cache_dir, frames_dir, train_ids)
    print("assembling test ...")
    Xte, yte, vof = _assemble(cache_dir, frames_dir, test_ids)
    print(f"train frames {len(ytr)}  test frames {len(yte)}\n")

    print("linear probe ...")
    lin = fit_linear_probe(Xtr, ytr, Xte, yte, PHASES, vof)
    print("kNN probe ...")
    knn = fit_knn_probe(Xtr, ytr, Xte, yte, k=args.k)

    print("video-level bootstrap ...")
    boot = _video_bootstrap(test_ids, lin.test_pred, lin.test_true, vof)

    print("\n" + "=" * 56)
    enc_name = Path(args.cache).parts[-3] if len(Path(args.cache).parts) >= 3 else "?"
    print(f"REPRODUCTION GATE  |  {enc_name} frozen  |  frame-level")
    print("=" * 56)
    print(f"{'':20} {'accuracy':>10} {'macro-F1':>10}")
    print(f"{'linear probe':20} {lin.accuracy:>10.3f} {lin.macro_f1:>10.3f}")
    print(f"{'  95% CI (video)':20} "
          f"[{boot['acc_ci'][0]:.3f},{boot['acc_ci'][1]:.3f}] "
          f"[{boot['f1_ci'][0]:.3f},{boot['f1_ci'][1]:.3f}]")
    print(f"{'kNN (k=%d)' % args.k:20} {knn['accuracy']:>10.3f} {knn['macro_f1']:>10.3f}")
    print(f"{'linear bal-acc':20} {lin.balanced_accuracy:>10.3f}")
    print("\nper-phase F1 (linear):")
    for ph, f in sorted(lin.per_phase_f1.items(), key=lambda x: -x[1]):
        print(f"  {ph:26} {f:.3f}")

    print("\ninterpretation:")
    if lin.accuracy > 0.85:
        print("  !! accuracy >0.85 frame-level -- SUSPECT LEAKAGE. Verify the")
        print("     split and the frame/label alignment before believing this.")
    elif lin.accuracy < 0.45:
        print("  !! accuracy <0.45 -- below frozen-feature expectation. Suspect")
        print("     a preprocessing mismatch (normalization, resize) or bad labels.")
    else:
        print("  OK: frozen-feature neighborhood. The gap below 90%+ temporal")
        print("      models is the project's premise, not a failure.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({
        "protocol": args.protocol,
        "linear": {"accuracy": lin.accuracy, "macro_f1": lin.macro_f1,
                   "balanced_accuracy": lin.balanced_accuracy,
                   "per_phase_f1": lin.per_phase_f1},
        "knn": knn,
        "bootstrap_video_level": boot,
        "n_train_frames": int(len(ytr)), "n_test_frames": int(len(yte)),
    }, indent=2))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
