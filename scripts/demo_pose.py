"""Demo: pose + segmentation overlay on a video file (default), webcam, or image.

Usage:
    python scripts/demo_pose.py                                 # default sample video
    python scripts/demo_pose.py --source data/raw/videos/squat.mp4
    python scripts/demo_pose.py --source 0                      # live webcam
    python scripts/demo_pose.py --source photo.jpg --image
    python scripts/demo_pose.py --no-blur                       # disable background blur
    python scripts/demo_pose.py --record out.mp4

Fetch sample clips first: `python scripts/fetch_videos.py`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Allow running as a plain script without installing the package.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitcoach.pose import PoseDetector
from fitcoach.drawing import draw_skeleton, blur_background, draw_fps


DEFAULT_SOURCE = str(ROOT / "data" / "raw" / "videos" / "sample.mp4")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FitCoach pose demo (Day 2)")
    p.add_argument(
        "--source",
        default=DEFAULT_SOURCE,
        help="path to a video/image, or webcam index like '0'. "
             "Default: data/raw/videos/sample.mp4 (run scripts/fetch_videos.py first).",
    )
    p.add_argument("--image", action="store_true", help="treat source as a still image")
    p.add_argument("--no-blur", action="store_true", help="disable background blur")
    p.add_argument("--no-skeleton", action="store_true", help="disable skeleton overlay")
    p.add_argument("--record", type=str, default=None, help="write output to mp4")
    p.add_argument("--complexity", type=int, default=1, choices=(0, 1, 2))
    return p.parse_args()


def open_source(src: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src!r}")
    return cap


def run_image(path: str, *, blur: bool, skeleton: bool, complexity: int) -> int:
    frame = cv2.imread(path)
    if frame is None:
        raise SystemExit(f"Could not read image: {path}")
    with PoseDetector(static_image_mode=True, model_complexity=complexity) as det:
        result = det.process(frame)
    if blur:
        frame = blur_background(frame, result)
    if skeleton:
        frame = draw_skeleton(frame, result)
    print(f"landmarks_found={result.found}")
    cv2.imshow("FitCoach — pose (image)", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return 0


def run_stream(src: str, *, blur: bool, skeleton: bool, complexity: int, record: str | None) -> int:
    cap = open_source(src)
    writer: cv2.VideoWriter | None = None
    if record is not None:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(record, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    ema_fps = 0.0
    alpha = 0.1
    last_t = time.perf_counter()

    with PoseDetector(model_complexity=complexity) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = det.process(frame)
            if blur:
                frame = blur_background(frame, result)
            if skeleton:
                frame = draw_skeleton(frame, result)

            now = time.perf_counter()
            inst_fps = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            ema_fps = inst_fps if ema_fps == 0 else (1 - alpha) * ema_fps + alpha * inst_fps
            draw_fps(frame, ema_fps)

            cv2.imshow("FitCoach — pose (q to quit)", frame)
            if writer is not None:
                writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    args = parse_args()
    if args.image:
        return run_image(
            args.source,
            blur=not args.no_blur,
            skeleton=not args.no_skeleton,
            complexity=args.complexity,
        )
    return run_stream(
        args.source,
        blur=not args.no_blur,
        skeleton=not args.no_skeleton,
        complexity=args.complexity,
        record=args.record,
    )


if __name__ == "__main__":
    raise SystemExit(main())
