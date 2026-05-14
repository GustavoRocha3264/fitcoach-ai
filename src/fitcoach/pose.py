"""Pose detection + body segmentation via MediaPipe Tasks API.

MediaPipe 0.10.35 removed the legacy `mp.solutions` namespace; this module
wraps the new `mp.tasks.vision.PoseLandmarker` so the rest of the project can
stay framework-agnostic. Returns a `PoseResult` carrying normalized landmarks
(xyz in [0,1]) and an optional segmentation mask.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as _mp_python
from mediapipe.tasks.python import vision as _mp_vision


# 33-landmark skeleton (BlazePose topology). Same set the legacy
# `mp.solutions.pose.POSE_CONNECTIONS` used to expose.
POSE_CONNECTIONS: frozenset[tuple[int, int]] = frozenset({
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (27, 29), (27, 31), (29, 31),
    (24, 26), (26, 28), (28, 30), (28, 32), (30, 32),
})

# Apple-public model variants. `heavy` is the most accurate, `lite` the
# fastest; `full` is the historical default. The variant determines the
# .task filename and the upstream URL.
MODEL_VARIANTS: dict[str, str] = {
    "lite":  "pose_landmarker_lite.task",
    "full":  "pose_landmarker_full.task",
    "heavy": "pose_landmarker_heavy.task",
}

_MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def model_path_for(variant: str) -> Path:
    """Resolve a variant name ("lite" / "full" / "heavy") to its on-disk path."""
    if variant not in MODEL_VARIANTS:
        raise ValueError(
            f"Unknown model variant {variant!r}. "
            f"Known: {sorted(MODEL_VARIANTS)}"
        )
    return _MODELS_DIR / MODEL_VARIANTS[variant]


_DEFAULT_MODEL = model_path_for("full")


@dataclass(frozen=True)
class Landmark:
    x: float           # normalized to image width  [0, 1]
    y: float           # normalized to image height [0, 1]
    z: float           # depth relative to hips, in image-width units
    visibility: float  # [0, 1]


@dataclass(frozen=True)
class PoseResult:
    landmarks: Optional[list[Landmark]]   # 33 landmarks if a person was found, else None
    segmentation_mask: Optional[np.ndarray]  # float32 HxW in [0,1], or None
    image_shape: tuple[int, int]          # (height, width) of the input frame

    @property
    def found(self) -> bool:
        return self.landmarks is not None


class PoseDetector:
    """Wrapper around MediaPipe Tasks `PoseLandmarker`.

    Use as a context manager so the underlying graph is released:

        with PoseDetector() as det:
            result = det.process(frame_bgr)

    `static_image_mode=True` selects RunningMode.IMAGE (no temporal tracking),
    appropriate for one-off photos. The default uses RunningMode.VIDEO.
    """

    def __init__(
        self,
        *,
        model_variant: str | None = None,
        model_path: str | Path | None = None,
        static_image_mode: bool = False,
        enable_segmentation: bool = True,
        num_poses: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_pose_presence_confidence: float = 0.5,
        # accepted for API parity with the legacy wrapper; unused in Tasks API
        model_complexity: int = 1,
    ) -> None:
        del model_complexity  # model variant is chosen via `model_variant` / `model_path`
        if model_path is not None and model_variant is not None:
            raise ValueError("Pass either model_variant or model_path, not both.")
        if model_path is None:
            model_path = model_path_for(model_variant) if model_variant else _DEFAULT_MODEL
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Pose model not found at {model_path}. Download with:\n"
                "  curl -L -o models/pose_landmarker_full.task "
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
                "pose_landmarker_full/float16/latest/pose_landmarker_full.task"
            )

        running_mode = (
            _mp_vision.RunningMode.IMAGE if static_image_mode else _mp_vision.RunningMode.VIDEO
        )
        options = _mp_vision.PoseLandmarkerOptions(
            base_options=_mp_python.BaseOptions(model_asset_path=str(model_path)),
            running_mode=running_mode,
            num_poses=num_poses,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_pose_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_segmentation_masks=enable_segmentation,
        )
        self._landmarker = _mp_vision.PoseLandmarker.create_from_options(options)
        self._video_mode = not static_image_mode
        self._t0 = time.perf_counter()

    def process(self, frame_bgr: np.ndarray) -> PoseResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        if self._video_mode:
            ts_ms = int((time.perf_counter() - self._t0) * 1000)
            raw = self._landmarker.detect_for_video(mp_image, ts_ms)
        else:
            raw = self._landmarker.detect(mp_image)

        landmarks: Optional[list[Landmark]] = None
        if raw.pose_landmarks:
            first = raw.pose_landmarks[0]
            landmarks = [Landmark(lm.x, lm.y, lm.z, lm.visibility) for lm in first]

        mask: Optional[np.ndarray] = None
        if raw.segmentation_masks:
            mask = raw.segmentation_masks[0].numpy_view()

        h, w = frame_bgr.shape[:2]
        return PoseResult(landmarks=landmarks, segmentation_mask=mask, image_shape=(h, w))

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "PoseDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
