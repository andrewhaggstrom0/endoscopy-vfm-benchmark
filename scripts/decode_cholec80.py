"""Decode Cholec80 to 1fps JPEG frames + a per-video label CSV.

Sequential read (not per-frame seek) -- decoding 80 full surgeries by seeking
each 25th frame would thrash. We read straight through and keep every 25th.

Output:
    $BIGDIR/endoscopy/raw/cholec80_frames/videoNN/frame_XXXXXX.jpg
    $BIGDIR/endoscopy/raw/cholec80_frames/videoNN/labels.csv

Idempotent: a video whose labels.csv exists and matches its frame count is
skipped. Safe to resubmit.

Run under tmux on a compute node:
    python scripts/decode_cholec80.py            # all 80
    python scripts/decode_cholec80.py --limit 2  # smoke test
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data import cholec80  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=f"{os.environ['BIGDIR']}/endoscopy/raw/cholec80")
    ap.add_argument("--out", default=f"{os.environ['BIGDIR']}/endoscopy/raw/cholec80_frames")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    root, out = Path(args.root), Path(args.out)
    vids = cholec80.video_ids(root)[: args.limit]
    print(f"{len(vids)} videos -> {out}\n")

    for i, vid in enumerate(vids, 1):
        labels = cholec80.align_labels(root, vid)
        vdir = out / vid
        csv = vdir / "labels.csv"
        if csv.exists():
            existing = len(list(vdir.glob("frame_*.jpg")))
            if existing == len(labels):
                print(f"[{i}/{len(vids)}] {vid}: {existing} frames, skip")
                continue
        vdir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(root / "videos" / f"{vid}.mp4"))
        want = dict(zip(labels["native_frame"], labels["sample_idx"]))
        saved, idx = 0, 0
        while cap.isOpened() and saved < len(want):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in want:
                fp = vdir / f"frame_{idx:06d}.jpg"
                cv2.imwrite(str(fp), frame,  # BGR on disk; loader converts on read
                            [cv2.IMWRITE_JPEG_QUALITY, args.quality])
                saved += 1
            idx += 1
        cap.release()

        if saved != len(labels):
            print(f"[{i}/{len(vids)}] {vid}: WARNING decoded {saved}/{len(labels)}")
        labels.to_csv(csv, index=False)
        dist = labels["phase"].value_counts().to_dict()
        print(f"[{i}/{len(vids)}] {vid}: {saved} frames | phases {dist}")

    print("\ndone. verify one video's alignment before extracting embeddings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
