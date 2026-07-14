"""Face detection backed by MediaPipe."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

_matplotlib_cache = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import cv2
import mediapipe as mp
import numpy as np

from face_monitor.utils.image_utils import to_rgb


@dataclass(frozen=True, slots=True)
class FaceBox:
    x1: int
    y1: int
    x2: int
    y2: int
    confidence: float

    @property
    def as_tuple(self) -> tuple[int, int, int, int]:
        return self.x1, self.y1, self.x2, self.y2


class FaceDetector:
    """Face detector returning pixel-space bounding boxes.

    Uses MediaPipe Solutions when available. Newer Python 3.13 MediaPipe wheels
    expose only the Tasks API, so OpenCV Haar cascades provide a reliable local
    fallback for server startup and basic monitoring.
    """

    def __init__(self, min_confidence: float) -> None:
        self._min_confidence = min_confidence
        self._detector = None
        self._cascade = None
        if hasattr(mp, "solutions"):
            self._detector = mp.solutions.face_detection.FaceDetection(
                model_selection=0,
                min_detection_confidence=min_confidence,
            )
        else:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(str(cascade_path))

    def detect(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        height, width = frame_bgr.shape[:2]
        if self._detector is None:
            return self._detect_with_opencv(frame_bgr)

        result = self._detector.process(to_rgb(frame_bgr))
        if not result.detections:
            return []

        faces: list[FaceBox] = []
        for detection in result.detections:
            bbox = detection.location_data.relative_bounding_box
            x1 = int(bbox.xmin * width)
            y1 = int(bbox.ymin * height)
            x2 = int((bbox.xmin + bbox.width) * width)
            y2 = int((bbox.ymin + bbox.height) * height)
            score = float(detection.score[0]) if detection.score else 0.0
            faces.append(FaceBox(x1, y1, x2, y2, score))
        return faces

    def _detect_with_opencv(self, frame_bgr: np.ndarray) -> list[FaceBox]:
        if self._cascade is None or self._cascade.empty():
            return []
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        return [
            FaceBox(int(x), int(y), int(x + w), int(y + h), self._min_confidence)
            for x, y, w, h in detections
        ]
