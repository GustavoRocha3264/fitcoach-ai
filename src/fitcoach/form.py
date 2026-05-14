"""Rule-based form evaluator.

For Day 3 we score reps by depth alone — the minimum angle reached during the
down-phase must cross an exercise-specific threshold. More rules (back
alignment, knees-over-toes, etc.) come later once we trust the landmark
stream.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DepthRule:
    """A rep is "deep enough" if min_angle ≤ `max_angle`.

    Score scales from 0 at `max_angle` to 1 at `ideal_angle` (deeper than
    ideal also caps at 1).
    """
    max_angle: float       # threshold to pass
    ideal_angle: float     # full credit at or below this

    def evaluate(self, min_angle: float) -> tuple[bool, float]:
        passed = min_angle <= self.max_angle
        # Linear scoring: 1 at ideal, 0 at threshold, clamped.
        denom = max(self.max_angle - self.ideal_angle, 1e-6)
        score = (self.max_angle - min_angle) / denom
        score = max(0.0, min(1.0, score))
        return passed, score


@dataclass(frozen=True)
class FormVerdict:
    exercise: str
    passed: bool
    score: float                # 0..1
    summary: str
    failed_rules: list[str] = field(default_factory=list)


EXERCISE_RULES: dict[str, DepthRule] = {
    # Thresholds in degrees. Lower = deeper bend.
    "squat":  DepthRule(max_angle=100.0, ideal_angle=75.0),   # hip-knee-ankle
    "pushup": DepthRule(max_angle=110.0, ideal_angle=80.0),   # shoulder-elbow-wrist
    "curl":   DepthRule(max_angle=60.0,  ideal_angle=40.0),   # shoulder-elbow-wrist
}


def evaluate_rep(exercise: str, *, min_angle: float) -> FormVerdict:
    """Score a completed rep given the minimum joint angle it reached."""
    rule = EXERCISE_RULES[exercise]  # KeyError on unknown is intentional
    passed, score = rule.evaluate(min_angle)
    if passed:
        summary = f"{exercise}: good depth ({min_angle:.0f}° ≤ {rule.max_angle:.0f}°)"
        failed: list[str] = []
    else:
        summary = f"{exercise}: shallow depth ({min_angle:.0f}° > {rule.max_angle:.0f}°)"
        failed = ["depth"]
    return FormVerdict(
        exercise=exercise,
        passed=passed,
        score=score,
        summary=summary,
        failed_rules=failed,
    )
