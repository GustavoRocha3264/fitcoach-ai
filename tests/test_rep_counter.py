"""Tests for the FSM rep counter."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.rep_counter import RepCounter, RepEvent  # noqa: E402


def feed(counter: RepCounter, angles: list[float]) -> list[RepEvent]:
    events: list[RepEvent] = []
    for a in angles:
        evt = counter.update(a)
        if evt is not None:
            events.append(evt)
    return events


def test_counter_starts_at_zero() -> None:
    c = RepCounter(down_threshold=90.0, up_threshold=160.0)
    assert c.count == 0
    assert c.state == "up"


def test_full_cycle_counts_one_rep() -> None:
    c = RepCounter(down_threshold=90.0, up_threshold=160.0)
    # Start extended → bend down → extend → that's one rep.
    events = feed(c, [170, 150, 120, 85, 70, 110, 150, 170])
    assert c.count == 1
    assert len(events) == 1
    assert events[0].rep_index == 1
    assert events[0].min_angle == pytest.approx(70.0)


def test_partial_dip_does_not_count() -> None:
    c = RepCounter(down_threshold=90.0, up_threshold=160.0)
    # Goes down only to 100 (above down_threshold), then back up.
    feed(c, [170, 130, 110, 100, 130, 170])
    assert c.count == 0


def test_hysteresis_prevents_bounce() -> None:
    """Small jitter around either threshold must not trigger spurious reps."""
    c = RepCounter(down_threshold=90.0, up_threshold=160.0)
    # Hover around the up threshold; never crosses 90 → 0 reps.
    feed(c, [159, 161, 159, 161, 159, 161, 158])
    assert c.count == 0


def test_two_reps_in_sequence() -> None:
    c = RepCounter(down_threshold=90.0, up_threshold=160.0)
    feed(c, [170, 80, 170, 80, 170])
    assert c.count == 2


def test_tracks_min_angle_per_rep() -> None:
    c = RepCounter(down_threshold=90.0, up_threshold=160.0)
    events = feed(c, [170, 80, 60, 75, 170, 170, 85, 70, 170])
    assert [e.min_angle for e in events] == pytest.approx([60.0, 70.0])


def test_nan_values_are_ignored() -> None:
    c = RepCounter(down_threshold=90.0, up_threshold=160.0)
    feed(c, [170, float("nan"), 80, float("nan"), 170])
    assert c.count == 1


def test_invalid_thresholds_raise() -> None:
    import pytest as _pt
    with _pt.raises(ValueError):
        RepCounter(down_threshold=160.0, up_threshold=90.0)
    with _pt.raises(ValueError):
        RepCounter(down_threshold=100.0, up_threshold=100.0)
