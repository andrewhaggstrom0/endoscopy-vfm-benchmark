"""CholecSeg8k frame loader.

Clips, not videos
-----------------
CholecSeg8k is short clips sampled from scattered points in each surgery, NOT
continuous video. Frames are nested as:

    video01/video01_00080/frame_80_endo.png

Measured on the extracted data: video01's 1280 frames contain index gaps of
81, 1167, 11916, and 14380 -- the largest being ~10 minutes of surgery at
25fps. Concatenating a video's frames into one array therefore creates
"adjacent" pairs that are minutes apart, which would silently contaminate
every inter-frame drift measurement in this project.

Worse, frame indices alone cannot detect this: video09 shows zero index
discontinuities despite being multiple clips. The clip *directory* is the only
reliable contiguity signal, so it is the unit of extraction.

Splits remain at the VIDEO level (clips from one surgery are near-duplicates),
so every clip carries its parent video id.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

MASK_TOKENS = ("mask", "watershed", "color")
IMG_EXT = (".png", ".jpg", ".jpeg")


def _is_frame(p: Path) -> bool:
    stem = p.stem.lower()
    return p.suffix.lower() in IMG_EXT and not any(t in stem for t in MASK_TOKENS)


def _frame_index(p: Path) -> int:
    nums = re.findall(r"\d+", p.stem)
    return int(nums[-1]) if nums else 0


def parent_video(clip_id: str) -> str:
    """'video01__00080' -> 'video01'. Used for video-level splits."""
    return clip_id.split("__")[0]


def _clip_id(p: Path, root: Path) -> str:
    """Clip directory is the contiguity unit. Falls back to the video dir."""
    rel = p.relative_to(root)
    parts = rel.parts[:-1]
    video = next((x for x in parts if re.match(r"(?i)^video\d+$", x)), None)
    clip = next((x for x in parts if re.match(r"(?i)^video\d+_\d+$", x)), None)
    if video and clip:
        return f"{video.lower()}__{clip.split('_')[-1]}"
    if clip:
        return clip.lower().replace("_", "__", 1)
    if video:
        return f"{video.lower()}__all"
    return "__".join(parts).lower() or "unknown__all"


def discover(root: str | Path) -> dict[str, list[Path]]:
    """{clip_id: [frame paths sorted by index]}. One entry per contiguous clip."""
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"dataset root does not exist: {root}")

    groups: dict[str, list[Path]] = {}
    for p in root.rglob("*"):
        if p.is_file() and _is_frame(p):
            groups.setdefault(_clip_id(p, root), []).append(p)

    if not groups:
        raise RuntimeError(
            f"no frame images found under {root}. Run probe_layout() first."
        )
    for c in groups:
        groups[c].sort(key=_frame_index)
    return dict(sorted(groups.items()))


def assert_contiguous(groups: dict[str, list[Path]], max_gap: int = 1) -> None:
    """Every clip must have consecutive frame indices.

    This is the guarantee the temporal metrics depend on. Violations mean the
    grouping is wrong and drift numbers would be measuring scene cuts.
    """
    bad = {}
    for clip, paths in groups.items():
        idx = np.array([_frame_index(p) for p in paths])
        gaps = np.diff(idx)
        if len(gaps) and gaps.max() > max_gap:
            bad[clip] = sorted(set(gaps[gaps > max_gap].tolist()))[:5]
    if bad:
        raise ValueError(f"non-contiguous clips: {bad}")


def probe_layout(root: str | Path, n: int = 10) -> None:
    root = Path(root)
    imgs = [p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXT]
    frames = [p for p in imgs if _is_frame(p)]
    groups = discover(root)
    vids = {parent_video(c) for c in groups}

    print(f"root            {root}")
    print(f"images          {len(imgs)}")
    print(f"frames kept     {len(frames)}")
    print(f"masks dropped   {len(imgs) - len(frames)}")
    print(f"videos          {len(vids)}")
    print(f"clips           {len(groups)}")
    lens = [len(v) for v in groups.values()]
    print(f"frames/clip     min={min(lens)} max={max(lens)} "
          f"median={int(np.median(lens))}")
    print("\nsample clips:")
    for c, paths in list(groups.items())[:n]:
        idx = [_frame_index(p) for p in paths]
        print(f"  {c:24} n={len(paths):4} idx {idx[0]}..{idx[-1]} "
              f"video={parent_video(c)}")
    try:
        assert_contiguous(groups)
        print("\ncontiguity      OK (all clips consecutive)")
    except ValueError as e:
        print(f"\ncontiguity      FAILED: {e}")


def load_frames(paths: list[Path]) -> np.ndarray:
    """-> (B, H, W, 3) uint8 RGB. cv2 reads BGR; the conversion is required."""
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
