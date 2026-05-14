"""Rendering helpers: skeleton overlay and background blur from segmentation mask."""
from __future__ import annotations

import cv2
import numpy as np

from .pose import POSE_CONNECTIONS, PoseResult


_SKELETON_COLOR = (0, 255, 0)
_JOINT_COLOR = (0, 165, 255)
_VISIBILITY_THRESHOLD = 0.5


def draw_skeleton(frame_bgr: np.ndarray, result: PoseResult) -> np.ndarray:
    """Draw landmarks + connections in-place and return the frame."""
    if not result.found:
        return frame_bgr

    h, w = result.image_shape
    pts: list[tuple[int, int] | None] = []
    for lm in result.landmarks:  # type: ignore[union-attr]
        if lm.visibility < _VISIBILITY_THRESHOLD:
            pts.append(None)
        else:
            pts.append((int(lm.x * w), int(lm.y * h)))

    for a, b in POSE_CONNECTIONS:
        pa, pb = pts[a], pts[b]
        if pa is not None and pb is not None:
            cv2.line(frame_bgr, pa, pb, _SKELETON_COLOR, 2)

    for p in pts:
        if p is not None:
            cv2.circle(frame_bgr, p, 4, _JOINT_COLOR, -1)
    return frame_bgr


def blur_background(
    frame_bgr: np.ndarray,
    result: PoseResult,
    *,
    blur_ksize: int = 35,
    threshold: float = 0.5,
) -> np.ndarray:
    """Keep the person sharp, blur the background using the segmentation mask."""
    if result.segmentation_mask is None:
        return frame_bgr
    mask = (result.segmentation_mask > threshold).astype(np.float32)
    mask = mask[..., None]  # HxWx1 for broadcasting
    blurred = cv2.GaussianBlur(frame_bgr, (blur_ksize, blur_ksize), 0)
    return (frame_bgr * mask + blurred * (1.0 - mask)).astype(np.uint8)


def draw_fps(frame_bgr: np.ndarray, fps: float) -> np.ndarray:
    cv2.putText(
        frame_bgr,
        f"{fps:5.1f} FPS",
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame_bgr
