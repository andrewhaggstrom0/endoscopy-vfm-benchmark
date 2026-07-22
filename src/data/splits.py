"""Video-level splits. Never split at the frame level.

Frames within one surgery are near-duplicates: adjacent frames differ by
milliseconds, and even distant frames share the patient, anatomy, lighting,
camera, and operator. A random frame split therefore puts near-copies of test
frames into train, and probe accuracy inflates substantially -- a result that
looks strong and means nothing.

Every split in this project is by video. Clips inherit their parent video's
assignment (see cholecseg8k.parent_video), so no surgery ever straddles the
train/test boundary.

Protocol note
-------------
Cholec80's literature uses several conventions -- 40/40, 40/8/32, 32/8/40.
They are not interchangeable and numbers are not comparable across them. Pick
one, record it in the config, and state it in the report. `PROTOCOLS` below
holds the named variants so the choice is explicit rather than implicit in a
random seed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Named literature protocols: (n_train, n_val, n_test), assigned in video order.
PROTOCOLS = {
    "twinanda_40_40": (40, 0, 40),
    "train40_val8_test32": (40, 8, 32),
    "train32_val8_test40": (32, 8, 40),
}


@dataclass(frozen=True)
class Split:
    train: list[str]
    val: list[str]
    test: list[str]
    protocol: str

    def assignment(self) -> dict[str, str]:
        return ({v: "train" for v in self.train}
                | {v: "val" for v in self.val}
                | {v: "test" for v in self.test})

    def summary(self) -> str:
        return (f"{self.protocol}: {len(self.train)} train / "
                f"{len(self.val)} val / {len(self.test)} test")


def ordered_split(videos: list[str], protocol: str) -> Split:
    """Deterministic split in sorted video order -- the literature convention."""
    if protocol not in PROTOCOLS:
        raise KeyError(f"unknown protocol {protocol!r}; have {list(PROTOCOLS)}")
    n_tr, n_va, n_te = PROTOCOLS[protocol]
    vids = sorted(videos)
    if len(vids) != n_tr + n_va + n_te:
        raise ValueError(
            f"protocol {protocol!r} expects {n_tr + n_va + n_te} videos, "
            f"got {len(vids)}. Use random_split for a different-sized dataset."
        )
    return Split(vids[:n_tr], vids[n_tr:n_tr + n_va], vids[n_tr + n_va:], protocol)


def random_split(videos: list[str], fractions=(0.6, 0.15, 0.25),
                 seed: int = 0) -> Split:
    """Seeded random video split, for datasets without a standard protocol."""
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"fractions must sum to 1, got {sum(fractions)}")
    vids = sorted(videos)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(vids))
    n_tr = int(round(fractions[0] * len(vids)))
    n_va = int(round(fractions[1] * len(vids)))
    pick = lambda ix: sorted(vids[i] for i in ix)
    return Split(pick(perm[:n_tr]), pick(perm[n_tr:n_tr + n_va]),
                 pick(perm[n_tr + n_va:]), f"random(seed={seed})")


def assert_no_leakage(split: Split) -> None:
    """No video may appear in two subsets. Called before every probe."""
    tr, va, te = set(split.train), set(split.val), set(split.test)
    for a, b, na, nb in ((tr, va, "train", "val"), (tr, te, "train", "test"),
                         (va, te, "val", "test")):
        if a & b:
            raise ValueError(f"video leakage between {na} and {nb}: {sorted(a & b)}")
    if len(tr) + len(va) + len(te) != len(tr | va | te):
        raise ValueError("duplicate videos within a subset")


def split_units(unit_ids: list[str], split: Split, parent_fn) -> dict[str, list[str]]:
    """Map clip/shard ids to subsets via their parent video.

    Raises if any unit's parent is unassigned -- silently dropping data is
    worse than failing.
    """
    assign = split.assignment()
    out: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    missing = set()
    for uid in unit_ids:
        parent = parent_fn(uid)
        if parent not in assign:
            missing.add(parent)
            continue
        out[assign[parent]].append(uid)
    if missing:
        raise ValueError(f"units with unassigned parent videos: {sorted(missing)}")
    return out
