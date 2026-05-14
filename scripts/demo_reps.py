"""Demo: angle + rep count + form feedback overlaid on a sample video.

Usage:
    python scripts/demo_reps.py --exercise squat
    python scripts/demo_reps.py --exercise pushup
    python scripts/demo_reps.py --exercise curl
    python scripts/demo_reps.py --exercise squat --source data/raw/videos/squat.mp4

The exercise selects which joint angle is tracked and which form rule applies.
Default `--source` points to `data/raw/videos/<exercise>.mp4`.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fitcoach.pose import PoseDetector
from fitcoach.drawing import draw_skeleton, draw_fps
from fitcoach.angles import knee_angle, elbow_angle, best_side_angle
from fitcoach.rep_counter import RepCounter
from fitcoach.form import evaluate_rep, EXERCISE_RULES


# Joint-angle selector per exercise + FSM thresholds (degrees).
EXERCISES: dict[str, dict] = {
    "squat":  {"angle_of": "knee",  "down": 100.0, "up": 160.0},
    "pushup": {"angle_of": "elbow", "down": 110.0, "up": 160.0},
    "curl":   {"angle_of": "elbow", "down": 60.0,  "up": 150.0},
}


def angle_for(result, kind: str) -> float:
    if not result.found:
        return float("nan")
    if kind == "knee":
        return best_side_angle(
            knee_angle(result.landmarks, side="left"),
            knee_angle(result.landmarks, side="right"),
        )
    return best_side_angle(
        elbow_angle(result.landmarks, side="left"),
        elbow_angle(result.landmarks, side="right"),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FitCoach rep counter demo (Day 3)")
    p.add_argument("--exercise", choices=sorted(EXERCISES), required=True)
    p.add_argument("--source", default=None,
                   help="video path or webcam index (default: data/raw/videos/<exercise>.mp4)")
    p.add_argument("--record", type=str, default=None)
    p.add_argument("--complexity", type=int, default=1, choices=(0, 1, 2))
    p.add_argument("--headless", action="store_true",
                   help="run without a display window (still prints rep events)")
    return p.parse_args()


def open_source(src: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(int(src)) if src.isdigit() else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise SystemExit(f"Could not open source: {src!r}")
    return cap


def overlay_hud(frame, *, angle: float, reps: int, exercise: str, last_msg: str) -> None:
    import math
    h = frame.shape[0]
    a_str = f"{angle:.0f}°" if not math.isnan(angle) else "—"
    cv2.putText(frame, f"{exercise.upper()}  angle: {a_str}", (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, f"reps: {reps}", (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    if last_msg:
        cv2.putText(frame, last_msg, (10, h - 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)


def main() -> int:
    args = parse_args()
    cfg = EXERCISES[args.exercise]
    source = args.source or str(ROOT / "data" / "raw" / "videos" / f"{args.exercise}.mp4")

    cap = open_source(source)
    writer: cv2.VideoWriter | None = None
    if args.record:
        fps_in = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(args.record, cv2.VideoWriter_fourcc(*"mp4v"), fps_in, (w, h))

    counter = RepCounter(down_threshold=cfg["down"], up_threshold=cfg["up"])
    last_verdict_msg = ""
    print(f"Tracking {args.exercise} via {cfg['angle_of']} angle "
          f"(down<{cfg['down']:.0f}°, up>{cfg['up']:.0f}°). Rule: "
          f"depth≤{EXERCISE_RULES[args.exercise].max_angle:.0f}°.")

    ema_fps = 0.0
    last_t = time.perf_counter()

    with PoseDetector(model_complexity=args.complexity) as det:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            result = det.process(frame)
            angle = angle_for(result, cfg["angle_of"])
            event = counter.update(angle)
            if event is not None:
                verdict = evaluate_rep(args.exercise, min_angle=event.min_angle)
                last_verdict_msg = ("✓ " if verdict.passed else "✗ ") + verdict.summary
                print(f"rep #{event.rep_index}: min_angle={event.min_angle:.1f}°  "
                      f"score={verdict.score:.2f}  {verdict.summary}")

            draw_skeleton(frame, result)
            now = time.perf_counter()
            inst = 1.0 / max(now - last_t, 1e-6)
            last_t = now
            ema_fps = inst if ema_fps == 0 else 0.9 * ema_fps + 0.1 * inst
            draw_fps(frame, ema_fps)
            overlay_hud(frame, angle=angle, reps=counter.count,
                        exercise=args.exercise, last_msg=last_verdict_msg)

            if writer is not None:
                writer.write(frame)
            if not args.headless:
                cv2.imshow("FitCoach — Day 3 (q to quit)", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    print(f"Total reps: {counter.count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
