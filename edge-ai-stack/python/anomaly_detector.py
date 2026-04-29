"""Anomaly detection logic for motor vibration streams."""

import math
from collections import deque


class AnomalyDetector:
    """Detects vibration anomalies with threshold and rolling statistics."""

    def __init__(self, threshold: float = 0.6, window_size: int = 10) -> None:
        """Initialize detector settings.

        Args:
            threshold: Absolute vibration threshold in g.
            window_size: Number of recent samples used for rolling statistics.
        """
        self.threshold = threshold
        self.window_size = window_size
        self.window: deque[float] = deque(maxlen=window_size)

    def detect(self, value: float) -> str:
        """Classify vibration value as NORMAL or FAULT.

        Rules:
        1) If value exceeds fixed threshold, classify as FAULT.
        2) If value exceeds rolling mean + 3*std, classify as FAULT.
        If either rule triggers, return FAULT.

        Args:
            value: Current vibration reading in g.

        Returns:
            "NORMAL" or "FAULT".
        """
        threshold_fault = value > self.threshold

        stats = self._compute_stats()
        std = stats["std"]
        mean = stats["mean"]
        statistical_fault = std > 0 and value > (mean + (3 * std))

        self.window.append(value)

        if threshold_fault or statistical_fault:
            return "FAULT"
        return "NORMAL"
    
    def get_stats(self) -> dict:
        """Return rolling statistical context for observability/debugging."""
        stats = self._compute_stats()
        return {
            "mean": stats["mean"],
            "std": stats["std"],
            "window": list(self.window),
            "threshold": self.threshold,
        }

    def _compute_stats(self) -> dict:
        """Compute mean/std for current rolling window (before latest append)."""
        if not self.window:
            return {"mean": 0.0, "std": 0.0}

        values = list(self.window)
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        std = math.sqrt(variance)
        return {"mean": round(mean, 6), "std": round(std, 6)}

   