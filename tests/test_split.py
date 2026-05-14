"""Tests for clip-level train/val split helpers.

Splitting windows directly would leak between train and val (windows from
the same clip are highly correlated). The helper must split at the clip
level instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.dataset import ClipEntry  # noqa: E402
from fitcoach.split import (  # noqa: E402
    split_entries_by_suffix,
    split_entries_by_count,
)


def _e(name: str, exercise: str) -> ClipEntry:
    return ClipEntry(path=Path(f"/tmp/{name}.npy"), exercise=exercise)


def _entries() -> list[ClipEntry]:
    return [
        _e("squat", "squat"), _e("squat_2", "squat"),
        _e("squat_3", "squat"), _e("squat_4", "squat"),
        _e("pushup", "pushup"), _e("pushup_2", "pushup"),
        _e("pushup_3", "pushup"), _e("pushup_4", "pushup"),
        _e("curl", "curl"), _e("curl_2", "curl"),
        _e("curl_3", "curl"), _e("curl_4", "curl"),
    ]


def test_split_by_suffix_holds_out_highest_per_class() -> None:
    train, val = split_entries_by_suffix(_entries(), val_suffix="_4")
    val_names = sorted(e.path.stem for e in val)
    assert val_names == ["curl_4", "pushup_4", "squat_4"]
    # Train and val are disjoint and together cover everything.
    train_names = sorted(e.path.stem for e in train)
    assert len(train_names) == 9
    assert set(train_names).isdisjoint(val_names)


def test_split_by_suffix_works_with_other_suffix() -> None:
    train, val = split_entries_by_suffix(_entries(), val_suffix="_2")
    val_names = sorted(e.path.stem for e in val)
    assert val_names == ["curl_2", "pushup_2", "squat_2"]
    assert len(train) == 9


def test_split_by_suffix_errors_on_missing_class() -> None:
    """If a class has no matching suffix, raise — silently leaving a class out
    of validation is exactly the kind of footgun TDD should prevent."""
    entries = [_e("squat", "squat"), _e("pushup_4", "pushup")]
    with pytest.raises(ValueError, match="squat"):
        split_entries_by_suffix(entries, val_suffix="_4")


def test_split_by_count_picks_n_per_class_deterministically() -> None:
    train, val = split_entries_by_count(_entries(), n_per_class=1, seed=0)
    # Each class gets exactly one val clip.
    assert sorted(e.exercise for e in val) == ["curl", "pushup", "squat"]
    assert len(train) == 9
    # Same seed ⇒ same selection.
    train2, val2 = split_entries_by_count(_entries(), n_per_class=1, seed=0)
    assert [e.path.stem for e in val] == [e.path.stem for e in val2]


def test_split_by_count_uses_different_seeds() -> None:
    _, v1 = split_entries_by_count(_entries(), n_per_class=1, seed=0)
    _, v2 = split_entries_by_count(_entries(), n_per_class=1, seed=42)
    # Very likely different — if not, the seeds aren't doing anything.
    assert {e.path.stem for e in v1} != {e.path.stem for e in v2}


def test_split_by_count_rejects_too_many() -> None:
    """Asking for more val clips than a class has must error."""
    with pytest.raises(ValueError):
        split_entries_by_count(_entries(), n_per_class=5, seed=0)
