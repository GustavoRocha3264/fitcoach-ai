"""Tests for the PyTorch LandmarkDataset."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitcoach.dataset import (  # noqa: E402
    EXERCISE_LABELS,
    LandmarkDataset,
    ClipEntry,
    scan_landmarks_dir,
)


def _write_clip(path: Path, n_frames: int, *, fill: float = 0.5) -> None:
    arr = np.full((n_frames, 33, 4), fill, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, arr)


def test_exercise_labels_are_stable() -> None:
    """Class IDs must be deterministic so saved models stay valid."""
    assert EXERCISE_LABELS == {"squat": 0, "pushup": 1, "curl": 2}


def test_windows_count_matches_stride_math(tmp_path: Path) -> None:
    clip = tmp_path / "a.npy"
    _write_clip(clip, n_frames=50)
    entries = [ClipEntry(path=clip, exercise="squat")]
    ds = LandmarkDataset(entries, window_size=30, stride=10)
    # 50 frames, w=30, s=10 → starts at 0, 10, 20 → 3 windows
    assert len(ds) == 3


def test_getitem_returns_tensor_and_label(tmp_path: Path) -> None:
    clip = tmp_path / "a.npy"
    _write_clip(clip, n_frames=40)
    ds = LandmarkDataset(
        [ClipEntry(path=clip, exercise="pushup")],
        window_size=30, stride=10,
    )
    x, y = ds[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape == (30, 33, 4)
    assert x.dtype == torch.float32
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.long
    assert y.item() == EXERCISE_LABELS["pushup"]


def test_clip_shorter_than_window_is_skipped(tmp_path: Path) -> None:
    short = tmp_path / "short.npy"
    _write_clip(short, n_frames=10)
    long = tmp_path / "long.npy"
    _write_clip(long, n_frames=40)
    ds = LandmarkDataset(
        [ClipEntry(path=short, exercise="squat"), ClipEntry(path=long, exercise="curl")],
        window_size=30, stride=10,
    )
    # short clip yields 0 windows; long clip yields 2 (starts 0, 10)
    assert len(ds) == 2
    _, y = ds[0]
    assert y.item() == EXERCISE_LABELS["curl"]


def test_nan_handling_fills_zeros(tmp_path: Path) -> None:
    """NaN values (from undetected frames) must be replaced before the network sees them."""
    clip = tmp_path / "a.npy"
    arr = np.full((40, 33, 4), 0.5, dtype=np.float32)
    arr[5] = np.nan
    np.save(clip, arr)
    ds = LandmarkDataset(
        [ClipEntry(path=clip, exercise="squat")],
        window_size=30, stride=10, nan_fill=0.0,
    )
    x, _ = ds[0]
    assert not torch.isnan(x).any()


def test_dataloader_iterates(tmp_path: Path) -> None:
    """Sanity: the dataset works with torch.utils.data.DataLoader."""
    for ex, name in (("squat", "a.npy"), ("pushup", "b.npy")):
        _write_clip(tmp_path / name, n_frames=60)
    entries = [
        ClipEntry(path=tmp_path / "a.npy", exercise="squat"),
        ClipEntry(path=tmp_path / "b.npy", exercise="pushup"),
    ]
    ds = LandmarkDataset(entries, window_size=30, stride=15)
    loader = torch.utils.data.DataLoader(ds, batch_size=2, shuffle=False)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (2, 30, 33, 4)
    assert batch_y.shape == (2,)


def test_scan_landmarks_dir_uses_filename_as_exercise(tmp_path: Path) -> None:
    """Files like data/processed/landmarks/squat.npy → exercise='squat'."""
    _write_clip(tmp_path / "squat.npy", 50)
    _write_clip(tmp_path / "pushup.npy", 50)
    _write_clip(tmp_path / "curl.npy", 50)
    _write_clip(tmp_path / "sample.npy", 50)  # not in EXERCISE_LABELS → skipped
    entries = scan_landmarks_dir(tmp_path)
    names = sorted(e.exercise for e in entries)
    assert names == ["curl", "pushup", "squat"]


def test_unknown_exercise_raises() -> None:
    with pytest.raises(KeyError):
        LandmarkDataset(
            [ClipEntry(path=Path("nope.npy"), exercise="backflip")],
            window_size=30, stride=10,
        )
