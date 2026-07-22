"""Cholec80 loader.

Distribution layout (raw video release, camma_public S3):
    videos/videoNN.mp4                     -- full surgery, 25 fps
    phase_annotations/videoNN-phase.txt    -- per-frame phase, 25 fps, TSV
    tool_annotations/videoNN-tool.txt      -- tool presence, 1 fps (every 25th frame)

Sampling protocol
-----------------
Standard Cholec80 evaluation samples at 1 fps = every 25th native frame.
Phase labels are at 25 fps, so the label for sampled frame k is phase[25*k].
Tool labels are already at 1 fps and keyed by native frame index, so they are
looked up directly.

Getting the stride wrong produces plausible-looking but systematically shifted
labels -- the canonical reproduction-gate failure. `align_labels` therefore
asserts index agreement rather than assuming it.

7 phases, 7 tools. Videos indexed video01..video80. Splits are video-level
(see splits.py); a single surgery never straddles train/test.
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

NATIVE_FPS = 25
SAMPLE_FPS = 1
STRIDE = NATIVE_FPS // SAMPLE_FPS  # 25

PHASES = [
    "Preparation", "CalotTriangleDissection", "ClippingCutting",
    "GallbladderDissection", "GallbladderPackaging",
    "CleaningCoagulation", "GallbladderRetraction",
]
PHASE_TO_IDX = {p: i for i, p in enumerate(PHASES)}

TOOLS = ["Grasper", "Bipolar", "Hook", "Scissors",
         "Clipper", "Irrigator", "SpecimenBag"]


def video_ids(root: str | Path) -> list[str]:
    root = Path(root)
    vids = sorted(p.stem for p in (root / "videos").glob("video*.mp4"))
    if not vids:
        raise RuntimeError(f"no videos under {root/'videos'}")
    return vids


def _read_phase(root: Path, vid: str) -> pd.DataFrame:
    df = pd.read_csv(root / "phase_annotations" / f"{vid}-phase.txt", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df  # Frame (25fps), Phase


def _read_tool(root: Path, vid: str) -> pd.DataFrame:
    df = pd.read_csv(root / "tool_annotations" / f"{vid}-tool.txt", sep="\t")
    df.columns = [c.strip() for c in df.columns]
    return df  # Frame (1fps), then 7 tool columns


def align_labels(root: str | Path, vid: str) -> pd.DataFrame:
    """Return one row per sampled (1 fps) frame with phase and tool labels.

    Columns: native_frame, sample_idx, phase (str), phase_idx (int),
    and one 0/1 column per tool.
    """
    root = Path(root)
    phase = _read_phase(root, vid)
    tool = _read_tool(root, vid)

    tool_frames = tool["Frame"].to_numpy()
    # The tool file *defines* the sampled frames. Every tool frame index must
    # exist in the 25fps phase file, or the two annotations disagree.
    if not np.all(tool_frames % STRIDE == 0):
        bad = tool_frames[tool_frames % STRIDE != 0][:5]
        raise ValueError(f"{vid}: tool frames not on {STRIDE}-stride: {bad}")

    phase_by_frame = dict(zip(phase["Frame"], phase["Phase"]))
    rows = []
    for k, nf in enumerate(tool_frames):
        if nf not in phase_by_frame:
            raise ValueError(f"{vid}: tool frame {nf} absent from phase file")
        ph = phase_by_frame[nf].strip()
        if ph not in PHASE_TO_IDX:
            raise ValueError(f"{vid}: unknown phase {ph!r}")
        row = {"native_frame": int(nf), "sample_idx": k,
               "phase": ph, "phase_idx": PHASE_TO_IDX[ph]}
        for t in TOOLS:
            row[t] = int(tool.iloc[k][t])
        rows.append(row)
    return pd.DataFrame(rows)


def decode_frames(root: str | Path, vid: str, native_frames: list[int]) -> np.ndarray:
    """Decode the requested native-frame indices -> (N, H, W, 3) uint8 RGB.

    Seeks by index. cv2 gives BGR; conversion is required or every encoder
    sees swapped channels and looks weak rather than broken.
    """
    root = Path(root)
    cap = cv2.VideoCapture(str(root / "videos" / f"{vid}.mp4"))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {vid}.mp4")
    want = set(native_frames)
    out, idx = {}, 0
    try:
        while cap.isOpened() and len(out) < len(want):
            ok, frame = cap.read()
            if not ok:
                break
            if idx in want:
                out[idx] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            idx += 1
    finally:
        cap.release()
    missing = want - set(out)
    if missing:
        raise RuntimeError(f"{vid}: could not decode frames {sorted(missing)[:5]}")
    return np.stack([out[f] for f in native_frames])
