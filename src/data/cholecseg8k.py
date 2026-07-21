"""CholecSeg8k frame loader.

Deliberately discovers structure by globbing rather than hardcoding a layout.
Public dataset directory conventions vary by mirror and by how the archive was
unpacked, and a hardcoded path that silently matches zero files is worse than
one that fails loudly. `probe_layout()` exists so you can confirm what was
actually found before extracting anything.

Frames are grouped by *video*, because every split in this project is at the
video level. Frames from one procedure are near-duplicates; a random frame
split leaks between train and test and inflates probe accuracy substantially.
Grouping here makes the correct split the path of least resistance later.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

# Mask files sit beside frames; exclude them or you encode annotations as data.
MASK_TOKENS = ("mask", "watershed", "color")
IMG_EXT = (".png", ".jpg", ".jpeg")


def _is_frame(p: Path) -> bool:
    stem = p.stem.lower()
    return p.suffix.lower() in IMG_EXT and not any(t in stem for t in MASK_TOKENS)


def _video_id(p: Path, root: Path) -> str:
    """First path component under root that looks like a video directory."""
    rel = p.relative_to(root)
    for part in rel.parts:
        if re.match(r"(?i)^video\d+", part):
            return part.lower()
    return rel.parts[0] if len(rel.parts) > 1 else "unknown"


def _frame_index(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def discover(root: str | Path) -> dict[str, list[Path]]:
    """{video_id: [frame paths sorted by frame index]}."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    groups: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and _is_frame(p):
            groups.setdefault(_video_id(p, root), []).append(p)

    if not groups:
        raise RuntimeError(
            f"no frame images found under {root}. Run probe_layout() to see "
            f"what is actually there before adjusting the filters."
        )
    for v in groups:
        groups[v].sort(key=_frame_index)
    return dict(sorted(groups.items()))


def probe_layout(root: str | Path, n: int = 20) -> None:
    """Print what discovery sees. Run this before your first extraction."""
    root = Path(root)
    files = [p for p in root.rglob("*") if p.is_file()]
    imgs = [p for p in files if p.suffix.lower() in IMG_EXT]
    frames = [p for p in imgs if _is_frame(p)]
    print(f"root          {root}")
    print(f"files         {len(files)}")
    print(f"images        {len(imgs)}")
    print(f"frames (kept) {len(frames)}")
    print(f"masks (dropped) {len(imgs) - len(frames)}")
    print("\nsample kept:")
    for p in frames[:n]:
        print(f"  {p.relative_to(root)}   -> video={_video_id(p, root)} idx={_frame_index(p)}")
    if len(imgs) != len(frames):
        print("\nsample dropped:")
        for p in [q for q in imgs if not _is_frame(q)][:5]:
            print(f"  {p.relative_to(root)}")


def load_frames(paths: list[Path]) -> np.ndarray:
    """-> (B, H, W, 3) uint8 RGB.

    cv2 reads BGR; the conversion is not optional. Feeding BGR to an
    ImageNet-normalized encoder degrades features in a way that looks like a
    weak model rather than a bug.
    """
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
