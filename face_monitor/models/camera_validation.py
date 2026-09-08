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
            session.metadata["frozen_reference_frame"] = gray
            return False
        abs_diff = cv2.absdiff(gray, session.last_frame_gray)
        difference = float(np.mean(abs_diff))
        max_diff = float(np.max(abs_diff))
        session.last_frame_gray = gray

        # A truly frozen camera feed (software crash/spoof) will have virtually identical frames (difference ~ 0).
        # A stationary user with a good webcam might have a mean difference of 0.5 - 2.5 due to sensor noise.
        # We enforce a strict threshold (<= 0.2 mean, <= 1.0 max) to ensure we NEVER falsely flag a live, stationary user.
        if difference <= 0.2 and max_diff <= 1.0:
            if session.freeze_started_at is None:
                session.freeze_started_at = now
                session.metadata["frozen_reference_frame"] = gray
            else:
                ref_gray = session.metadata.get("frozen_reference_frame")
                if ref_gray is not None:
                    ref_diff = cv2.absdiff(gray, ref_gray)
                    ref_mean = float(np.mean(ref_diff))
                    ref_max = float(np.max(ref_diff))
                    # If over time they move enough (e.g. blink, breathe), max diff spikes
                    if ref_mean > self._freeze_difference_threshold or ref_max > 15.0:
                        session.freeze_started_at = None
                        return False

            return now - session.freeze_started_at >= self._freeze_seconds

        session.freeze_started_at = None
        return False
