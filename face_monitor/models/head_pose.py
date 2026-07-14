"""Head pose estimation using MediaPipe Face Mesh landmarks."""

from __future__ import annotations

import os
from pathlib import Path

_matplotlib_cache = Path(__file__).resolve().parents[2] / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import cv2
import mediapipe as mp
import numpy as np

from face_monitor.utils.image_utils import to_rgb


class HeadPoseEstimator:
    """Estimates coarse gaze direction from face mesh landmarks."""

    _LANDMARK_IDS = (1, 152, 33, 263, 61, 291)
    _MODEL_POINTS = np.array(
        [
            (0.0, 0.0, 0.0),
            (0.0, -63.6, -12.5),
            (-43.3, 32.7, -26.0),
            (43.3, 32.7, -26.0),
            (-28.9, -28.9, -24.1),
            (28.9, -28.9, -24.1),
        ],
        dtype=np.float64,
    )

    def __init__(self) -> None:
        self._mesh = None
        self._cascade = None
        if hasattr(mp, "solutions"):
            self._mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        else:
            cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
            self._cascade = cv2.CascadeClassifier(str(cascade_path))

    def estimate(self, frame_bgr: np.ndarray) -> str:
        if self._mesh is None:
            return self._estimate_with_opencv(frame_bgr)

        height, width = frame_bgr.shape[:2]
        result = self._mesh.process(to_rgb(frame_bgr))
        if not result.multi_face_landmarks:
            return "unknown"

        landmarks = result.multi_face_landmarks[0].landmark
        image_points = np.array(
            [(landmarks[idx].x * width, landmarks[idx].y * height) for idx in self._LANDMARK_IDS],
            dtype=np.float64,
        )
        focal_length = width
        camera_matrix = np.array(
            [[focal_length, 0, width / 2], [0, focal_length, height / 2], [0, 0, 1]],
            dtype=np.float64,
        )
        success, rotation_vector, _translation_vector = cv2.solvePnP(
            self._MODEL_POINTS,
            image_points,
            camera_matrix,
            np.zeros((4, 1)),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return "unknown"

        rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
        angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
        pitch, yaw, _roll = angles

        if yaw < -18:
            return "left"
        if yaw > 18:
            return "right"
        if pitch < -15:
            return "down"
        return "screen"

    def _estimate_with_opencv(self, frame_bgr: np.ndarray) -> str:
        if self._cascade is None or self._cascade.empty():
            return "unknown"
        height, width = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(60, 60),
        )
        if len(faces) == 0:
            return "unknown"

        x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
        center_x = x + w / 2
        center_y = y + h / 2
        if center_x < width * 0.35:
            return "left"
        if center_x > width * 0.65:
            return "right"
        if center_y > height * 0.68:
            return "down"
        return "screen"
