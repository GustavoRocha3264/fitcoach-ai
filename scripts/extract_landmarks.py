"""Extract landmark sequences from every video in data/raw/videos/.

Outputs go to data/processed/landmarks/<stem>.npy with shape (T, 33, 4).

Usage:
    python scripts/extract_landmarks.py
    python scripts/extract_landmarks.py --videos data/raw/videos --out data/processed/landmarks
    python scripts/extract_landmarks.py --max-frames 200       # quick sanity run
    python scripts/extract_landmarks.py --force                # re-extract existing
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitcoach.extract import extract_landmarks_from_video, save_landmarks  # noqa: E402
from fitcoach.dataset import exercise_from_stem  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract landmark sequences from videos")
    p.add_argument("--videos", type=Path, default=ROOT / "data" / "raw" / "videos")
    p.add_argument("--out", type=Path, default=ROOT / "data" / "processed" / "landmarks")
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--force", action="store_true", help="overwrite existing .npy files")
    p.add_argument(
        "--include-unknown",
        action="store_true",
        help="extract every .mp4, not just files whose stem matches a known exercise",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    videos = sorted(args.videos.glob("*.mp4"))
    if not videos:
        print(f"No mp4 files in {args.videos}")
        return 1

    for video in videos:
        if not args.include_unknown and exercise_from_stem(video.stem) is None:
            print(f"  · skip {video.name} (not a known exercise)")
            continue
        target = args.out / f"{video.stem}.npy"
        if target.exists() and not args.force:
            print(f"  · skip {video.name} (already extracted)")
            continue

        t0 = time.perf_counter()
        arr = extract_landmarks_from_video(video, max_frames=args.max_frames)
        save_landmarks(arr, target)
        dt = time.perf_counter() - t0
        import numpy as np
        n_detected = int(np.isfinite(arr[:, 0, 0]).sum())
        print(
            f"  → {video.name}: {arr.shape[0]} frames "
            f"({n_detected} with pose)  {dt:.1f}s  → {target.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
