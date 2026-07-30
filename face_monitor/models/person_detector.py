"""Person detection backed by OpenCV."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from face_monitor.utils.image_utils import to_rgb

class PersonDetector:
    """Detects if a person's body is present using MediaPipe Pose."""

    def __init__(self) -> None:
        self._pose = None
        if hasattr(mp, "solutions"):
            self._pose = mp.solutions.pose.Pose(
                static_image_mode=False,
                model_complexity=0, # Use fastest model
                min_detection_confidence=0.4,
            )
        else:
            # Fallback if mediapipe solutions is somehow not available
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_upperbody.xml"
            self._cascade = cv2.CascadeClassifier(str(cascade_path))

    def detect(self, frame_bgr: np.ndarray) -> bool:
        if self._pose is not None:
            result = self._pose.process(to_rgb(frame_bgr))
            return result.pose_landmarks is not None
            
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
