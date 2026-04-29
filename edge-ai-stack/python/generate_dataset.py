"""Generate a synthetic labelled dataset for motor vibration anomaly detection.

This script uses the existing vibration simulator to create realistic training
data with engineered features suitable for machine-learning models.
"""

import csv
import os
import random
from collections import deque

from vibration_simulator import generate_vibration

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASET_DIR = os.path.join(os.path.dirname(__file__), "dataset")
DATASET_PATH = os.path.join(DATASET_DIR, "vibration_data.csv")

TOTAL_SAMPLES = 10_000
FAULT_RATIO = 0.15  # ~15 % anomalous samples
WINDOW_SIZE = 10  # sliding-window for rolling statistics

FIELDNAMES = [
    "vibration",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "delta",
    "label",
]


# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

def compute_rolling_mean(window: deque) -> float:
    """Return the arithmetic mean of values in *window*."""
    return sum(window) / len(window)


def compute_rolling_std(window: deque, mean: float) -> float:
    """Return the population standard deviation of *window*."""
    if len(window) < 2:
        return 0.0
    variance = sum((x - mean) ** 2 for x in window) / len(window)
    return variance ** 0.5


def compute_features(window: deque, current: float, previous: float) -> dict:
    """Compute all engineered features for one sample.

    Args:
        window: Recent vibration readings (including *current*).
        current: The latest vibration reading.
        previous: The immediately preceding vibration reading.

    Returns:
        A dict of feature-name → value ready for CSV writing.
    """
    r_mean = compute_rolling_mean(window)
    r_std = compute_rolling_std(window, r_mean)
    r_min = min(window)
    r_max = max(window)
    delta = abs(current - previous)

    return {
        "vibration": round(current, 4),
        "rolling_mean": round(r_mean, 4),
        "rolling_std": round(r_std, 4),
        "rolling_min": round(r_min, 4),
        "rolling_max": round(r_max, 4),
        "delta": round(delta, 4),
    }


# ---------------------------------------------------------------------------
# Main generation routine
# ---------------------------------------------------------------------------

def generate_dataset(
    total_samples: int = TOTAL_SAMPLES,
    fault_ratio: float = FAULT_RATIO,
    window_size: int = WINDOW_SIZE,
    output_path: str = DATASET_PATH,
) -> str:
    """Generate a CSV dataset and return the file path.

    The dataset has *total_samples* rows.  Approximately *fault_ratio* of them
    are fault (label = 1) and the rest are normal (label = 0).

    Args:
        total_samples: Number of rows to generate.
        fault_ratio: Proportion of fault samples (0.0 – 1.0).
        window_size: Sliding window length for rolling statistics.
        output_path: Destination CSV path.

    Returns:
        Absolute path to the written CSV file.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    window: deque[float] = deque(maxlen=window_size)
    previous = 0.0
    rows_written = 0

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=FIELDNAMES)
        writer.writeheader()

        for _ in range(total_samples):
            is_fault = random.random() < fault_ratio
            vibration = generate_vibration(fault=is_fault)

            window.append(vibration)

            # Only start writing once the window is full so features are stable
            if len(window) < window_size:
                previous = vibration
                continue

            features = compute_features(window, vibration, previous)
            features["label"] = 1 if is_fault else 0
            writer.writerow(features)
            rows_written += 1
            previous = vibration

    print(f"Dataset generated: {rows_written} rows -> {output_path}")
    return output_path


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    generate_dataset()
