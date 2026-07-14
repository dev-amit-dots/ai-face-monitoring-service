"""Session-local face embedding and comparison."""

from __future__ import annotations

import cv2
import numpy as np

from face_monitor.models.face_detector import FaceBox
from face_monitor.utils.image_utils import clamp_box

try:
    import face_recognition as face_recognition_lib
except ImportError:  # pragma: no cover - optional production dependency path.
    face_recognition_lib = None


class FaceRecognizer:
    """Creates embeddings using face_recognition when available.

    The fallback embedding is a normalized HSV and texture histogram. It is not
    a biometric-grade recognizer, but keeps the service operational in minimal
    CPU-only deployments until a stronger embedding backend is installed.
    """

    def __init__(self, tolerance: float) -> None:
        self._tolerance = tolerance

    def embedding(self, frame_bgr: np.ndarray, face: FaceBox) -> np.ndarray | None:
        if face_recognition_lib is not None:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            top, right, bottom, left = face.y1, face.x2, face.y2, face.x1
            encodings = face_recognition_lib.face_encodings(
                rgb, known_face_locations=[(top, right, bottom, left)]
            )
            if encodings:
                return np.asarray(encodings[0], dtype=np.float32)

        return self._fallback_embedding(frame_bgr, face)

    def is_match(self, registered: np.ndarray, current: np.ndarray) -> bool:
        distance = float(np.linalg.norm(registered - current))
        return distance <= self._tolerance

    @staticmethod
    def _fallback_embedding(frame_bgr: np.ndarray, face: FaceBox) -> np.ndarray | None:
        height, width = frame_bgr.shape[:2]
        x1, y1, x2, y2 = clamp_box(face.as_tuple, width, height)
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        crop = cv2.resize(crop, (96, 96), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256])
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        texture = cv2.Laplacian(gray, cv2.CV_32F).reshape(-1)
        texture_hist, _ = np.histogram(texture, bins=32, range=(-64, 64), density=True)
        vector = np.concatenate([hist.flatten(), texture_hist.astype(np.float32)])
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return (vector / norm).astype(np.float32)

