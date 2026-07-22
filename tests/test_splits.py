"""Leakage tests. These run in CI and gate every probe."""

from __future__ import annotations

import pytest

from src.data.splits import (PROTOCOLS, Split, assert_no_leakage, ordered_split,
                             random_split, split_units)

VIDS80 = [f"video{i:02d}" for i in range(1, 81)]


def test_ordered_split_sizes():
    s = ordered_split(VIDS80, "twinanda_40_40")
    assert (len(s.train), len(s.val), len(s.test)) == (40, 0, 40)
    assert_no_leakage(s)


@pytest.mark.parametrize("proto", list(PROTOCOLS))
def test_all_protocols_are_leak_free(proto):
    assert_no_leakage(ordered_split(VIDS80, proto))


def test_ordered_split_is_deterministic():
    assert ordered_split(VIDS80, "twinanda_40_40") == ordered_split(
        list(reversed(VIDS80)), "twinanda_40_40")


def test_wrong_video_count_rejected():
    with pytest.raises(ValueError, match="expects 80"):
        ordered_split(VIDS80[:17], "twinanda_40_40")


def test_random_split_covers_all_videos_once():
    s = random_split(VIDS80, seed=7)
    assert_no_leakage(s)
    assert sorted(s.train + s.val + s.test) == VIDS80


def test_random_split_seed_reproducible():
    assert random_split(VIDS80, seed=3) == random_split(VIDS80, seed=3)
    assert random_split(VIDS80, seed=3) != random_split(VIDS80, seed=4)


def test_leakage_is_detected():
    bad = Split(train=["v1", "v2"], val=["v2"], test=["v3"], protocol="bad")
    with pytest.raises(ValueError, match="leakage between train and val"):
        assert_no_leakage(bad)


def test_clips_follow_parent_video():
    """The property that makes clip-level shards safe."""
    s = Split(train=["video01"], val=[], test=["video02"], protocol="t")
    clips = ["video01__00080", "video01__00160", "video02__00080"]
    out = split_units(clips, s, lambda c: c.split("__")[0])
    assert out["train"] == ["video01__00080", "video01__00160"]
    assert out["test"] == ["video02__00080"]
    assert not set(out["train"]) & set(out["test"])


def test_unassigned_parent_raises():
    s = Split(train=["video01"], val=[], test=[], protocol="t")
    with pytest.raises(ValueError, match="unassigned"):
        split_units(["video99__00080"], s, lambda c: c.split("__")[0])
