"""Tests for the landmark-extraction pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitcoach.extract import extract_landmarks, save_landmarks  # noqa: E402
from fitcoach.pose import PoseDetector, PoseResult, Landmark  # noqa: E402


class _FakeDetector:
    """Returns a canned sequence of PoseResults — fast, no MediaPipe."""

    def __init__(self, results: list[PoseResult]) -> None:
        self._results = list(results)
        self._i = 0

    def process(self, frame_bgr) -> PoseResult:
        r = self._results[self._i]
        self._i += 1
        return r


def _result(found: bool, xy: float = 0.5, vis: float = 1.0) -> PoseResult:
    if not found:
        return PoseResult(landmarks=None, segmentation_mask=None, image_shape=(10, 10))
    lms = [Landmark(x=xy, y=xy, z=0.0, visibility=vis) for _ in range(33)]
    return PoseResult(landmarks=lms, segmentation_mask=None, image_shape=(10, 10))


def test_extract_shape_is_T_33_4() -> None:
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(4)]
    det = _FakeDetector([_result(True), _result(True), _result(True), _result(True)])
    arr = extract_landmarks(frames, det)
    assert arr.shape == (4, 33, 4)
    assert arr.dtype == np.float32


def test_extract_fills_nan_for_undetected_frames() -> None:
    frames = [np.zeros((10, 10, 3), dtype=np.uint8) for _ in range(3)]
    det = _FakeDetector([_result(True), _result(False), _result(True)])
    arr = extract_landmarks(frames, det)
    assert not np.isnan(arr[0]).any()
    assert np.isnan(arr[1]).all()
    assert not np.isnan(arr[2]).any()


def test_extract_carries_visibility() -> None:
    det = _FakeDetector([_result(True, vis=0.3)])
    arr = extract_landmarks([np.zeros((10, 10, 3), dtype=np.uint8)], det)
    assert arr[0, 0, 3] == pytest.approx(0.3)


def test_save_landmarks_round_trip(tmp_path: Path) -> None:
    arr = np.arange(4 * 33 * 4, dtype=np.float32).reshape(4, 33, 4)
    out = tmp_path / "clip.npy"
    save_landmarks(arr, out)
    assert out.exists()
    loaded = np.load(out)
    np.testing.assert_array_equal(loaded, arr)


def test_extract_with_real_detector_on_blank_frames() -> None:
    """Real MediaPipe on blank frames: shape is right, all landmarks NaN."""
    frames = [np.zeros((120, 160, 3), dtype=np.uint8) for _ in range(2)]
    with PoseDetector(static_image_mode=True, model_complexity=0) as det:
        arr = extract_landmarks(frames, det)
    assert arr.shape == (2, 33, 4)
    assert np.isnan(arr).all()
