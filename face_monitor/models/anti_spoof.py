"""Anti-spoofing model wrapper with heuristic fallback."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - optional runtime dependency path.
    ort = None


class AntiSpoofDetector:
    """Detects obvious spoof attempts.

    If ``SPOOF_MODEL_PATH`` points to an ONNX binary classifier, that model is
    used. Otherwise the detector falls back to conservative blur/edge heuristics
    that catch many printed-photo and screen-replay attempts without blocking
    normal users aggressively.
    """

    def __init__(self, model_path: str | None, threshold: float) -> None:
        self._threshold = threshold
        self._session = None
        self._input_name = None
        if model_path and ort is not None and Path(model_path).exists():
            self._session = ort.InferenceSession(
                model_path, providers=["CPUExecutionProvider"]
            )
            self._input_name = self._session.get_inputs()[0].name

    def is_spoof(self, frame_bgr: np.ndarray) -> bool:
        if self._session is not None and self._input_name is not None:
            score = self._predict_spoof_score(frame_bgr)
            return score >= self._threshold
        return self._heuristic_spoof(frame_bgr)

    def _predict_spoof_score(self, frame_bgr: np.ndarray) -> float:
        image = cv2.resize(frame_bgr, (224, 224), interpolation=cv2.INTER_AREA)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        tensor = np.transpose(image, (2, 0, 1))[None, ...]
        outputs = self._session.run(None, {self._input_name: tensor})
        return float(np.ravel(outputs[0])[-1])

    @staticmethod
    def _heuristic_spoof(frame_bgr: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        edges = cv2.Canny(gray, 80, 160)
        edge_density = float(np.count_nonzero(edges)) / float(edges.size)
        highlights = float(np.mean(gray > 245))

        very_flat_print = blur_score < 18 and edge_density < 0.025
        screen_glare = highlights > 0.18 and edge_density > 0.12
        return very_flat_print or screen_glare

