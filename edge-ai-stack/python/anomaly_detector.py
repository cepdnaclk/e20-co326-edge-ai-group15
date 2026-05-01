"""Runtime anomaly detector that wraps a trained Isolation Forest model.

This module is imported by ``main.py`` to perform real-time AI-based
anomaly detection on incoming vibration readings.
"""

import os
from collections import deque

import joblib
import numpy as np

# ---------------------------------------------------------------------------
# Default paths (relative to this file)
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(__file__)
DEFAULT_MODEL_PATH = os.path.join(_BASE_DIR, "model", "anomaly_detector.joblib")
DEFAULT_SCALER_PATH = os.path.join(_BASE_DIR, "model", "scaler.joblib")

DEFAULT_WINDOW_SIZE = 10


class AnomalyDetector:
    """Stateful anomaly detector backed by a trained Isolation Forest.

    The detector maintains a sliding window of recent vibration values so that
    it can compute the same rolling statistics used during training.

    Attributes:
        model: Trained ``IsolationForest`` instance.
        scaler: Fitted ``StandardScaler`` instance.
        window: Sliding window of recent vibration readings.
        previous: The most recent vibration value (for delta calculation).
        ready: Whether the window has been filled and predictions can begin.
    """

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        scaler_path: str = DEFAULT_SCALER_PATH,
        window_size: int = DEFAULT_WINDOW_SIZE,
    ) -> None:
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.window: deque[float] = deque(maxlen=window_size)
        self.window_size = window_size
        self.previous: float = 0.0
        self.ready: bool = False

        print(
            f"AnomalyDetector loaded (window={window_size}, "
            f"model={os.path.basename(model_path)})"
        )

    # ------------------------------------------------------------------
    # Feature computation (mirrors generate_dataset.py)
    # ------------------------------------------------------------------

    def _compute_features(self, vibration: float) -> np.ndarray:
        """Build the same feature vector used during training."""
        w = list(self.window)
        r_mean = sum(w) / len(w)
        variance = sum((x - r_mean) ** 2 for x in w) / len(w)
        r_std = variance ** 0.5
        r_min = min(w)
        r_max = max(w)
        delta = abs(vibration - self.previous)

        return np.array(
            [[vibration, r_mean, r_std, r_min, r_max, delta]],
            dtype=np.float64,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear runtime state so a restarted motor begins with a fresh window."""
        self.window.clear()
        self.previous = 0.0
        self.ready = False

    def predict(self, vibration: float) -> tuple[str, float]:
        """Classify a single vibration reading.

        Args:
            vibration: Latest vibration value in *g*.

        Returns:
            A ``(status, confidence)`` tuple where *status* is ``"FAULT"`` or
            ``"NORMAL"`` and *confidence* is a float between 0.0 and 1.0
            indicating how certain the model is about the prediction.
        """
        self.window.append(vibration)

        if len(self.window) < self.window_size:
            self.previous = vibration
            self.ready = False
            return ("NORMAL", 0.0)  # not enough data yet

        self.ready = True
        features = self._compute_features(vibration)
        scaled = self.scaler.transform(features)

        # Isolation Forest: decision_function returns anomaly score
        # More negative = more anomalous
        raw_score = self.model.decision_function(scaled)[0]
        prediction = self.model.predict(scaled)[0]  # 1 = inlier, -1 = outlier

        # Convert raw anomaly score to 0-1 confidence
        # decision_function: negative = anomaly, positive = normal
        # We use a sigmoid-like mapping for interpretability
        confidence = 1.0 / (1.0 + np.exp(5.0 * raw_score))  # higher when anomalous
        confidence = float(np.clip(confidence, 0.0, 1.0))

        status = "FAULT" if prediction == -1 else "NORMAL"
        self.previous = vibration

        return (status, round(confidence, 4))
