"""Turn a video (or any frame iterable) into a landmark tensor.

Output shape per clip: (T, 33, 4) float32 where channels are (x, y, z,
visibility). Frames with no detection are filled with NaN so downstream
code can mask or interpolate without losing temporal alignment.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Protocol

import numpy as np

from .pose import Landmark, PoseResult

_NUM_LANDMARKS = 33
_NUM_CHANNELS = 4  # x, y, z, visibility


class _Detector(Protocol):
    def process(self, frame_bgr: np.ndarray) -> PoseResult: ...


def _result_to_row(result: PoseResult) -> np.ndarray:
    if not result.found or result.landmarks is None:
        return np.full((_NUM_LANDMARKS, _NUM_CHANNELS), np.nan, dtype=np.float32)
    out = np.empty((_NUM_LANDMARKS, _NUM_CHANNELS), dtype=np.float32)
    lms: list[Landmark] = result.landmarks
    for i, lm in enumerate(lms):
        out[i] = (lm.x, lm.y, lm.z, lm.visibility)
    return out


def extract_landmarks(
    frames: Iterable[np.ndarray],
    detector: _Detector,
) -> np.ndarray:
    """Run `detector.process` over every frame and stack the results.

    Returns shape (T, 33, 4) float32. NaN-filled rows for undetected frames.
    """
    rows: list[np.ndarray] = []
    for frame in frames:
        rows.append(_result_to_row(detector.process(frame)))
    if not rows:
        return np.empty((0, _NUM_LANDMARKS, _NUM_CHANNELS), dtype=np.float32)
    return np.stack(rows, axis=0)


def extract_landmarks_from_video(
    video_path: str | Path,
    *,
    detector: _Detector | None = None,
    max_frames: int | None = None,
) -> np.ndarray:
    """OpenCV-backed wrapper around `extract_landmarks` for an mp4/etc."""
    import cv2  # local import keeps the module importable without cv2 in tests

    from .pose import PoseDetector

    own_detector = detector is None
    if detector is None:
        detector = PoseDetector(static_image_mode=False, enable_segmentation=False)

    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        def _frame_iter():
            n = 0
            while True:
                if max_frames is not None and n >= max_frames:
                    return
                ok, frame = cap.read()
                if not ok:
                    return
                n += 1
                yield frame

        try:
            return extract_landmarks(_frame_iter(), detector)
        finally:
            cap.release()
    finally:
        if own_detector and hasattr(detector, "close"):
            detector.close()


def save_landmarks(arr: np.ndarray, path: str | Path) -> None:
    """Persist a landmark tensor as .npy, creating parent dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)
