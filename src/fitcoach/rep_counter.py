"""FSM rep counter over a 1-D angle stream with hysteresis."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

State = Literal["up", "down"]


@dataclass(frozen=True)
class RepEvent:
    """Emitted exactly once per completed rep (on the down→up transition)."""
    rep_index: int   # 1-based
    min_angle: float


class RepCounter:
    """Counts repetitions from a 1-D joint-angle stream.

    Hysteresis is mandatory: `down_threshold < up_threshold`. Going below
    `down_threshold` arms a rep; going back above `up_threshold` completes it.
    NaN samples (occluded landmarks, dropouts) are ignored.

    Convention: extended joint = high angle (~170°), flexed = low angle.
    A "rep" is a full extend→flex→extend cycle.
    """

    def __init__(self, *, down_threshold: float, up_threshold: float) -> None:
        if not down_threshold < up_threshold:
            raise ValueError("down_threshold must be strictly less than up_threshold")
        self._down = down_threshold
        self._up = up_threshold
        self._state: State = "up"
        self._count = 0
        self._min_in_rep = math.inf  # smallest angle seen during current down-phase

    @property
    def count(self) -> int:
        return self._count

    @property
    def state(self) -> State:
        return self._state

    def update(self, angle: float) -> Optional[RepEvent]:
        """Feed one sample; return a RepEvent if a rep just completed, else None."""
        if math.isnan(angle):
            return None

        if self._state == "up":
            if angle < self._down:
                # Cross into down-phase: start tracking min.
                self._state = "down"
                self._min_in_rep = angle
            return None

        # state == "down"
        if angle < self._min_in_rep:
            self._min_in_rep = angle
        if angle > self._up:
            # Rep complete.
            self._count += 1
            evt = RepEvent(rep_index=self._count, min_angle=float(self._min_in_rep))
            self._state = "up"
            self._min_in_rep = math.inf
            return evt
        return None
