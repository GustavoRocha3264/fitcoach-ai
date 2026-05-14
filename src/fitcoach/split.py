"""Train/val split helpers — clip-level to avoid window-correlation leakage."""
from __future__ import annotations

import random
from collections import defaultdict

from .dataset import ClipEntry, EXERCISE_LABELS


def _group_by_class(entries: list[ClipEntry]) -> dict[str, list[ClipEntry]]:
    by_class: dict[str, list[ClipEntry]] = defaultdict(list)
    for e in entries:
        by_class[e.exercise].append(e)
    return by_class


def split_entries_by_suffix(
    entries: list[ClipEntry],
    *,
    val_suffix: str,
) -> tuple[list[ClipEntry], list[ClipEntry]]:
    """Hold out the clip whose stem ends with `val_suffix` (e.g. "_4") per class.

    Raises if any class has no matching clip — silently dropping a class from
    validation is exactly the kind of mistake that destroys reproducibility.
    """
    by_class = _group_by_class(entries)
    val: list[ClipEntry] = []
    train: list[ClipEntry] = []
    for cls in EXERCISE_LABELS:
        bucket = by_class.get(cls, [])
        match = [e for e in bucket if e.path.stem.endswith(val_suffix)]
        if not match:
            raise ValueError(
                f"class {cls!r} has no clip ending with {val_suffix!r}; "
                f"available stems: {[e.path.stem for e in bucket]}"
            )
        val.extend(match)
        train.extend(e for e in bucket if e not in match)
    return train, val


def split_entries_by_count(
    entries: list[ClipEntry],
    *,
    n_per_class: int,
    seed: int = 0,
) -> tuple[list[ClipEntry], list[ClipEntry]]:
    """Randomly hold out `n_per_class` clips per exercise. Deterministic per seed."""
    by_class = _group_by_class(entries)
    rng = random.Random(seed)
    val: list[ClipEntry] = []
    train: list[ClipEntry] = []
    for cls in sorted(by_class):  # deterministic iteration order
        bucket = list(by_class[cls])
        if n_per_class > len(bucket):
            raise ValueError(
                f"class {cls!r} has {len(bucket)} clip(s); cannot hold out {n_per_class}"
            )
        rng.shuffle(bucket)
        val.extend(bucket[:n_per_class])
        train.extend(bucket[n_per_class:])
    return train, val
