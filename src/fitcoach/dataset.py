"""PyTorch Dataset over per-frame landmark arrays.

Each landmark file is a (T, 33, 4) float32 .npy produced by `extract.py`.
The dataset slides a fixed-size window across each clip and emits
(window, label) pairs ready for a sequence model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


# Fixed class-id mapping. Order is part of the public contract — changing
# it would invalidate trained checkpoints.
EXERCISE_LABELS: dict[str, int] = {
    "squat": 0,
    "pushup": 1,
    "curl": 2,
}


@dataclass(frozen=True)
class ClipEntry:
    path: Path
    exercise: str


class LandmarkDataset(Dataset):
    """Windows a list of landmark clips into (window, label) tensors.

    Each clip is split into fixed-size windows of `window_size` frames with
    a hop of `stride` frames. Clips shorter than the window are dropped.
    NaN entries (frames where MediaPipe didn't detect a pose) are replaced
    with `nan_fill` before the tensor leaves the dataset, so models never
    see NaN.
    """

    def __init__(
        self,
        entries: list[ClipEntry],
        *,
        window_size: int = 30,
        stride: int = 15,
        nan_fill: float = 0.0,
    ) -> None:
        if window_size <= 0 or stride <= 0:
            raise ValueError("window_size and stride must be positive")
        self._window = window_size
        self._stride = stride
        self._nan_fill = nan_fill

        # Validate labels up front; cheap and avoids per-getitem surprises.
        for e in entries:
            if e.exercise not in EXERCISE_LABELS:
                raise KeyError(f"Unknown exercise label: {e.exercise!r}")

        # Pre-load arrays + precompute window index → (clip_idx, start).
        # Clips are small (<10k frames × 33 × 4 × 4 bytes ≈ 5 MB) so keeping
        # them resident is fine and avoids per-iter disk I/O.
        self._arrays: list[np.ndarray] = []
        self._labels: list[int] = []
        self._index: list[tuple[int, int]] = []
        for clip_idx, entry in enumerate(entries):
            arr = np.load(entry.path)
            if arr.ndim != 3 or arr.shape[1:] != (33, 4):
                raise ValueError(
                    f"{entry.path}: expected shape (T, 33, 4), got {arr.shape}"
                )
            self._arrays.append(arr.astype(np.float32, copy=False))
            self._labels.append(EXERCISE_LABELS[entry.exercise])
            if arr.shape[0] < window_size:
                continue
            last_start = arr.shape[0] - window_size
            for start in range(0, last_start + 1, stride):
                self._index.append((clip_idx, start))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        clip_idx, start = self._index[idx]
        window = self._arrays[clip_idx][start : start + self._window]
        window = np.nan_to_num(window, nan=self._nan_fill, copy=True)
        x = torch.from_numpy(window)
        y = torch.tensor(self._labels[clip_idx], dtype=torch.long)
        return x, y


def clip_coverage(arr: np.ndarray) -> float:
    """Fraction of frames where pose was detected.

    A frame counts as "detected" if its landmark row is finite (the extractor
    writes NaN for undetected frames). Empty clips report 0.0 so they're
    treated like fully-failed detections by filters.
    """
    if arr.shape[0] == 0:
        return 0.0
    # The extractor writes whole-row NaN for missed frames, so checking the
    # first landmark's x coordinate is enough to distinguish detected vs not.
    return float(np.isfinite(arr[:, 0, 0]).mean())


def exercise_from_stem(stem: str) -> str | None:
    """Map a filename stem to its exercise label, or None if not a known exercise.

    The prefix up to the first underscore is the label. So `squat`, `squat_2`,
    and `squat_alt` all map to `squat`. Stems whose prefix isn't in
    `EXERCISE_LABELS` (e.g. `sample`, `warmup_3`) return None.
    """
    if not stem:
        return None
    prefix = stem.split("_", 1)[0]
    return prefix if prefix in EXERCISE_LABELS else None


def scan_landmarks_dir(
    path: str | Path,
    *,
    min_coverage: float = 0.0,
) -> list[ClipEntry]:
    """Build entries from a directory whose filenames carry the exercise label.

    See `exercise_from_stem` for the naming convention. Files whose prefix
    isn't a known exercise are ignored. When `min_coverage > 0`, each
    candidate clip is loaded and dropped if its pose-detection rate is below
    the threshold (see `clip_coverage`).
    """
    out: list[ClipEntry] = []
    for p in sorted(Path(path).glob("*.npy")):
        label = exercise_from_stem(p.stem)
        if label is None:
            continue
        if min_coverage > 0.0:
            arr = np.load(p, mmap_mode="r")
            if clip_coverage(np.asarray(arr)) < min_coverage:
                continue
        out.append(ClipEntry(path=p, exercise=label))
    return out
