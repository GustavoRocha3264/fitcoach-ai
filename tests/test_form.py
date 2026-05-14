"""Tests for the rule-based form evaluator."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.form import FormVerdict, evaluate_rep, EXERCISE_RULES  # noqa: E402


def test_squat_good_depth_passes() -> None:
    v = evaluate_rep("squat", min_angle=85.0)
    assert v.passed is True
    assert "depth" in v.summary.lower() or "good" in v.summary.lower()


def test_squat_shallow_depth_fails() -> None:
    v = evaluate_rep("squat", min_angle=120.0)
    assert v.passed is False
    assert v.failed_rules == ["depth"]


def test_pushup_good_depth_passes() -> None:
    v = evaluate_rep("pushup", min_angle=80.0)
    assert v.passed is True


def test_pushup_shallow_fails() -> None:
    v = evaluate_rep("pushup", min_angle=140.0)
    assert v.passed is False


def test_curl_full_contraction_passes() -> None:
    v = evaluate_rep("curl", min_angle=45.0)
    assert v.passed is True


def test_curl_partial_fails() -> None:
    v = evaluate_rep("curl", min_angle=90.0)
    assert v.passed is False


def test_unknown_exercise_raises() -> None:
    import pytest
    with pytest.raises(KeyError):
        evaluate_rep("backflip", min_angle=10.0)


def test_verdict_is_dataclass_with_score() -> None:
    v = evaluate_rep("squat", min_angle=85.0)
    assert isinstance(v, FormVerdict)
    assert 0.0 <= v.score <= 1.0
    # Deeper than threshold should yield higher score than barely passing.
    deeper = evaluate_rep("squat", min_angle=70.0)
    assert deeper.score >= v.score


def test_rules_table_lists_known_exercises() -> None:
    assert {"squat", "pushup", "curl"} <= set(EXERCISE_RULES.keys())
