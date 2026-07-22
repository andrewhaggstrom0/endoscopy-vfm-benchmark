"""Loader for decoded Cholec80 frames (see scripts/decode_cholec80.py).

Layout:
    cholec80_frames/videoNN/frame_XXXXXX.jpg   (1 fps, native-frame-indexed)
    cholec80_frames/videoNN/labels.csv

Unlike CholecSeg8k, Cholec80's 1fps samples are a single continuous sequence
per surgery, so each video is one contiguous unit. The extract pipeline's
per-"clip" shard is therefore per-video here.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np


def _frame_index(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def parent_video(unit_id: str) -> str:
    return unit_id  # one unit == one video


def discover(root: str | Path) -> dict[str, list[Path]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    groups = {}
    for vdir in sorted(root.glob("video*")):
        if vdir.is_dir():
            fs = sorted(vdir.glob("frame_*.jpg"), key=_frame_index)
            if fs:
                groups[vdir.name] = fs
    if not groups:
        raise RuntimeError(f"no frame dirs under {root}")
    return groups


def assert_contiguous(groups, max_gap: int = 25) -> None:
    """1fps frames sit on a 25-native-frame stride; gaps beyond one step flag
    a decode hole. Not fatal to extraction, but recorded so drift metrics can
    exclude spanning pairs later."""
    bad = {}
    for vid, paths in groups.items():
        idx = np.array([_frame_index(p) for p in paths])
        gaps = np.diff(idx)
        over = gaps[gaps > max_gap]
        if len(over):
            bad[vid] = int(over.max())
    if bad:
        print(f"WARNING non-contiguous (decode holes): {bad}")


def load_frames(paths: list[Path]) -> np.ndarray:
    out = []
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"failed to read {p}")
        out.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    shapes = {a.shape for a in out}
    if len(shapes) > 1:
        raise ValueError(f"mixed frame shapes in one batch: {shapes}")
    return np.stack(out)
