"""Joint-angle math over landmark vectors. Pure NumPy."""
from __future__ import annotations

import math

import numpy as np

from .pose import Landmark


def joint_angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at `b` formed by segments b→a and b→c, in degrees ∈ [0, 180].

    Returns NaN if either segment has zero length.
    """
    ba = a - b
    bc = c - b
    na = float(np.linalg.norm(ba))
    nc = float(np.linalg.norm(bc))
    if na == 0.0 or nc == 0.0:
        return float("nan")
    cos = float(np.dot(ba, bc)) / (na * nc)
    cos = max(-1.0, min(1.0, cos))  # clamp before acos for numerical safety
    return math.degrees(math.acos(cos))


def landmark_angle(
    a: Landmark,
    b: Landmark,
    c: Landmark,
    *,
    min_visibility: float = 0.5,
) -> float:
    """Angle at landmark `b` using 2D (xy) coordinates.

    Returns NaN if any landmark's visibility is below `min_visibility`.
    """
    if min(a.visibility, b.visibility, c.visibility) < min_visibility:
        return float("nan")
    pa = np.array([a.x, a.y], dtype=np.float32)
    pb = np.array([b.x, b.y], dtype=np.float32)
    pc = np.array([c.x, c.y], dtype=np.float32)
    return joint_angle(pa, pb, pc)


# BlazePose 33-landmark indices for the joints we care about.
class L:
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28


def knee_angle(landmarks: list[Landmark], *, side: str = "left") -> float:
    """Hip–knee–ankle angle (extension ≈ 180°, deep squat ≈ 70–90°)."""
    if side == "left":
        h, k, a = L.LEFT_HIP, L.LEFT_KNEE, L.LEFT_ANKLE
    else:
        h, k, a = L.RIGHT_HIP, L.RIGHT_KNEE, L.RIGHT_ANKLE
    return landmark_angle(landmarks[h], landmarks[k], landmarks[a])


def elbow_angle(landmarks: list[Landmark], *, side: str = "left") -> float:
    """Shoulder–elbow–wrist angle (extension ≈ 180°, full curl ≈ 30–50°)."""
    if side == "left":
        s, e, w = L.LEFT_SHOULDER, L.LEFT_ELBOW, L.LEFT_WRIST
    else:
        s, e, w = L.RIGHT_SHOULDER, L.RIGHT_ELBOW, L.RIGHT_WRIST
    return landmark_angle(landmarks[s], landmarks[e], landmarks[w])


def best_side_angle(left: float, right: float) -> float:
    """Return whichever side angle is available (non-NaN); average if both."""
    left_ok = not math.isnan(left)
    right_ok = not math.isnan(right)
    if left_ok and right_ok:
        return (left + right) / 2.0
    if left_ok:
        return left
    if right_ok:
        return right
    return float("nan")
