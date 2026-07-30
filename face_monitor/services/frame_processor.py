"""Frame-level monitoring pipeline."""

from __future__ import annotations

import time
from dataclasses import dataclass

from face_monitor.config import Settings
from face_monitor.models.anti_spoof import AntiSpoofDetector
from face_monitor.models.camera_validation import CameraValidator
from face_monitor.models.face_detector import FaceDetector
from face_monitor.models.face_recognition import FaceRecognizer
from face_monitor.models.head_pose import HeadPoseEstimator
from face_monitor.models.person_detector import PersonDetector
from face_monitor.session_manager import SessionState


@dataclass(frozen=True, slots=True)
class MonitoringResult:
    status: str
    message: str
    details: dict[str, str | int | float | bool]


class FrameProcessor:
    """Coordinates camera checks, face AI, and status selection."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._camera = CameraValidator(
            settings.dark_mean_threshold,
            settings.dark_std_threshold,
            settings.freeze_seconds,
            settings.freeze_difference_threshold,
        )
        self._face_detector = FaceDetector(settings.face_detection_confidence)
        self._recognizer = FaceRecognizer(settings.face_match_tolerance)
        self._head_pose = HeadPoseEstimator()
        self._person_detector = PersonDetector()
        self._anti_spoof = AntiSpoofDetector(
            settings.spoof_model_path, settings.spoof_score_threshold
        )

    def process(self, session: SessionState, frame_bgr: np.ndarray) -> MonitoringResult:
        now = time.time()

        if self._camera.is_blocked(frame_bgr):
            return self._remember(session, "NOT_FOUND", "Camera appears blocked")

        if self._camera.is_frozen(session, frame_bgr, now):
            return self._remember(session, "NOT_FOUND", "Camera frame is frozen")

        faces = self._face_detector.detect(frame_bgr)
        if not faces:
            if self._person_detector.detect(frame_bgr):
                session.away_started_at = None
                return self._remember(
                    session, 
                    "FACE_PRESENT", 
                    "User is writing notes or face not visible"
                )
            
            if session.away_started_at is None:
                session.away_started_at = now
            if now - session.away_started_at >= self._settings.looking_away_seconds:
                return self._remember(session, "NOT_FOUND", "User not found")
            else:
                return self._remember(
                    session,
                    "FACE_PRESENT",
                    "User temporarily not visible"
                )
        
        if len(faces) > 1:
            session.away_started_at = None
            return self._remember(
                session,
                "FACE_PRESENT",
                "Multiple faces detected, but allowed",
                face_count=len(faces),
            )

        face = faces[0]

        # if self._anti_spoof.is_spoof(frame_bgr):
        #     return self._remember(session, "NOT_FOUND", "Potential spoof detected")

        embedding = self._recognizer.embedding(frame_bgr, face)
        if embedding is None:
            return self._remember(session, "NOT_FOUND", "Unable to create face embedding")

        # The user requested to disable face matching against the registered user
        # so we just record the embedding if needed but do not fail on mismatch.
        if session.registered_embedding is None:
            session.registered_embedding = embedding

        pose = self._head_pose.estimate(frame_bgr)
        # Face is present whether looking at camera or not
        session.away_started_at = None

        return self._remember(
            session,
            "FACE_PRESENT",
            "Face detected successfully",
            face_count=1,
            head_pose=pose,
        )

    @staticmethod
    def _remember(
        session: SessionState,
        status: str,
        message: str,
        **details: str | int | float | bool,
    ) -> MonitoringResult:
        session.last_detected_status = status
        session.warning_counters[status] = session.warning_counters.get(status, 0) + 1
        return MonitoringResult(status=status, message=message, details=details)
