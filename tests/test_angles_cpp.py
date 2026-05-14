"""Verify the C++ angle module matches the pure-Python implementation.

The C++ extension is built by `scripts/build_cpp.sh` (CMake + pybind11) into
`cpp/build/`. If it isn't built, every test in this module is skipped so the
rest of the suite stays green.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "cpp" / "build"))

try:
    import fitcoach_cpp  # type: ignore[import-not-found]
except ImportError:
    fitcoach_cpp = None  # noqa: F841 — flag for skip

from fitcoach.angles import joint_angle as joint_angle_py  # noqa: E402

pytestmark = pytest.mark.skipif(
    fitcoach_cpp is None,  # type: ignore[truthy-bool]
    reason="C++ extension not built — run scripts/build_cpp.sh",
)


def test_module_exposes_joint_angle() -> None:
    assert hasattr(fitcoach_cpp, "joint_angle")


@pytest.mark.parametrize("a,b,c,expected", [
    ((0.0, 1.0), (0.0, 0.0), (1.0, 0.0), 90.0),
    ((-1.0, 0.0), (0.0, 0.0), (1.0, 0.0), 180.0),
    ((1.0, 0.0), (0.0, 0.0), (1.0, 0.0), 0.0),
])
def test_known_angles_match_python(a, b, c, expected) -> None:
    got_cpp = fitcoach_cpp.joint_angle(a, b, c)
    got_py = joint_angle_py(np.array(a), np.array(b), np.array(c))
    assert got_cpp == pytest.approx(expected, abs=1e-4)
    assert got_cpp == pytest.approx(got_py, abs=1e-5)


def test_returns_nan_on_zero_length() -> None:
    got = fitcoach_cpp.joint_angle((0.0, 0.0), (0.0, 0.0), (1.0, 0.0))
    assert math.isnan(got)


def test_random_inputs_match_python_within_tolerance() -> None:
    rng = np.random.default_rng(42)
    for _ in range(100):
        a, b, c = rng.standard_normal((3, 2)).astype(np.float64)
        got_cpp = fitcoach_cpp.joint_angle(tuple(a), tuple(b), tuple(c))
        got_py = joint_angle_py(a, b, c)
        if math.isnan(got_py):
            assert math.isnan(got_cpp)
        else:
            assert got_cpp == pytest.approx(got_py, abs=1e-5)
