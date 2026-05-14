"""Unit tests for joint-angle math. Pure-NumPy, no MediaPipe needed."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.angles import joint_angle, landmark_angle  # noqa: E402
from fitcoach.pose import Landmark  # noqa: E402


def _pt(x: float, y: float) -> np.ndarray:
    return np.array([x, y], dtype=np.float32)


def test_right_angle() -> None:
    a, b, c = _pt(0, 1), _pt(0, 0), _pt(1, 0)
    assert joint_angle(a, b, c) == pytest.approx(90.0, abs=1e-4)


def test_straight_angle() -> None:
    a, b, c = _pt(-1, 0), _pt(0, 0), _pt(1, 0)
    assert joint_angle(a, b, c) == pytest.approx(180.0, abs=1e-4)


def test_zero_angle() -> None:
    a, b, c = _pt(1, 0), _pt(0, 0), _pt(1, 0)
    assert joint_angle(a, b, c) == pytest.approx(0.0, abs=1e-4)


def test_sixty_degrees() -> None:
    # Equilateral triangle vertices: angle at origin is 60°.
    a = _pt(1, 0)
    b = _pt(0, 0)
    c = _pt(math.cos(math.radians(60)), math.sin(math.radians(60)))
    assert joint_angle(a, b, c) == pytest.approx(60.0, abs=1e-4)


def test_works_in_3d() -> None:
    a = np.array([0, 1, 0], dtype=np.float32)
    b = np.array([0, 0, 0], dtype=np.float32)
    c = np.array([1, 0, 0], dtype=np.float32)
    assert joint_angle(a, b, c) == pytest.approx(90.0, abs=1e-4)


def test_returns_nan_on_zero_length_vector() -> None:
    a = _pt(0, 0)
    b = _pt(0, 0)  # coincident with a
    c = _pt(1, 0)
    assert math.isnan(joint_angle(a, b, c))


def test_landmark_angle_uses_xy() -> None:
    la = Landmark(x=0.5, y=0.2, z=0.0, visibility=1.0)
    lb = Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0)
    lc = Landmark(x=0.8, y=0.5, z=0.0, visibility=1.0)
    # Vertical segment + horizontal segment → 90°.
    assert landmark_angle(la, lb, lc) == pytest.approx(90.0, abs=1e-4)


def test_landmark_angle_returns_nan_when_low_visibility() -> None:
    la = Landmark(x=0.5, y=0.2, z=0.0, visibility=0.1)  # not visible
    lb = Landmark(x=0.5, y=0.5, z=0.0, visibility=1.0)
    lc = Landmark(x=0.8, y=0.5, z=0.0, visibility=1.0)
    assert math.isnan(landmark_angle(la, lb, lc, min_visibility=0.5))
