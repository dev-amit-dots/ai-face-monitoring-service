"""Camera health checks such as blocked and frozen video."""

from __future__ import annotations

import time

import cv2
import numpy as np

from face_monitor.session_manager import SessionState
from face_monitor.utils.image_utils import to_gray


class CameraValidator:
    def __init__(
        self,
        dark_mean_threshold: float,
        dark_std_threshold: float,
        freeze_seconds: float,
        freeze_difference_threshold: float,
    ) -> None:
        self._dark_mean_threshold = dark_mean_threshold
        self._dark_std_threshold = dark_std_threshold
        self._freeze_seconds = freeze_seconds
        self._freeze_difference_threshold = freeze_difference_threshold

    def is_blocked(self, frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        mean = float(np.mean(gray))
        std = float(np.std(gray))
        return mean < self._dark_mean_threshold and std < self._dark_std_threshold

    def is_frozen(self, session: SessionState, frame_bgr: np.ndarray, now: float) -> bool:
        gray = to_gray(frame_bgr)
        if session.last_frame_gray is None:
            session.last_frame_gray = gray
            session.freeze_started_at = None
            return False

        difference = float(np.mean(cv2.absdiff(gray, session.last_frame_gray)))
        session.last_frame_gray = gray

        if difference <= self._freeze_difference_threshold:
            if session.freeze_started_at is None:
                session.freeze_started_at = now
            return now - session.freeze_started_at >= self._freeze_seconds

        session.freeze_started_at = None
        return False
