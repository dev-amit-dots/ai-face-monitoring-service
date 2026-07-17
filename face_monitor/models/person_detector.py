"""Person detection backed by OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class PersonDetector:
    """Detects upper body of a person."""

    def __init__(self) -> None:
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_upperbody.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))

    def detect(self, frame_bgr: np.ndarray) -> bool:
        if self._cascade is None or self._cascade.empty():
            return False

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=3,
            minSize=(50, 50),
        )
        return len(detections) > 0
