"""Smoke tests for the pose module — no camera required."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fitcoach.pose import PoseDetector, PoseResult
from fitcoach.drawing import draw_skeleton, blur_background


def test_detector_runs_on_blank_frame() -> None:
    """Detector should process a blank frame without crashing and report no person."""
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with PoseDetector(static_image_mode=True, model_complexity=0) as det:
        result = det.process(frame)
    assert isinstance(result, PoseResult)
    assert result.image_shape == (480, 640)
    assert result.found is False
    assert result.landmarks is None


def test_drawing_is_a_noop_when_no_person() -> None:
    frame = np.zeros((120, 160, 3), dtype=np.uint8)
    result = PoseResult(landmarks=None, segmentation_mask=None, image_shape=(120, 160))
    out = draw_skeleton(frame.copy(), result)
    assert out.shape == frame.shape
    assert np.array_equal(out, frame)  # no drawing happened


def test_blur_background_handles_missing_mask() -> None:
    frame = np.full((80, 80, 3), 127, dtype=np.uint8)
    result = PoseResult(landmarks=None, segmentation_mask=None, image_shape=(80, 80))
    out = blur_background(frame, result)
    assert np.array_equal(out, frame)


def test_blur_background_applies_mask() -> None:
    h, w = 100, 100
    frame = np.full((h, w, 3), 255, dtype=np.uint8)  # white frame
    # Mask: keep left half (sharp), blur right half (still white -> still white)
    mask = np.zeros((h, w), dtype=np.float32)
    mask[:, : w // 2] = 1.0
    result = PoseResult(landmarks=None, segmentation_mask=mask, image_shape=(h, w))
    out = blur_background(frame, result)
    assert out.shape == frame.shape
    assert out.dtype == np.uint8
