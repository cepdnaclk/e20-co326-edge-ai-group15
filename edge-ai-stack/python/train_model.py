"""Train an Isolation Forest anomaly-detection model and persist it to disk.

Usage:
    python train_model.py            # uses default paths
    python train_model.py --dataset dataset/vibration_data.csv --output model/anomaly_detector.joblib
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(__file__)
DEFAULT_DATASET = os.path.join(BASE_DIR, "dataset", "vibration_data.csv")
DEFAULT_MODEL_DIR = os.path.join(BASE_DIR, "model")
DEFAULT_MODEL_PATH = os.path.join(DEFAULT_MODEL_DIR, "anomaly_detector.joblib")
DEFAULT_SCALER_PATH = os.path.join(DEFAULT_MODEL_DIR, "scaler.joblib")

FEATURE_COLUMNS = [
    "vibration",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "delta",
]
LABEL_COLUMN = "label"

# Isolation Forest parameters
CONTAMINATION = 0.15  # expected proportion of anomalies (matches simulator)
N_ESTIMATORS = 150
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Training logic
# ---------------------------------------------------------------------------

def train(
    dataset_path: str = DEFAULT_DATASET,
    model_path: str = DEFAULT_MODEL_PATH,
    scaler_path: str = DEFAULT_SCALER_PATH,
) -> dict:
    """Train the model and return evaluation metrics.

    Args:
        dataset_path: Path to the labelled CSV.
        model_path: Where to save the trained model.
        scaler_path: Where to save the fitted scaler.

    Returns:
        A dict containing train/test split sizes and the classification report.
    """
    # ---- Load data ----------------------------------------------------------
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset: {len(df)} rows, {len(df.columns)} columns")
    print(f"Label distribution:\n{df[LABEL_COLUMN].value_counts().to_string()}\n")

    X = df[FEATURE_COLUMNS].values
    y_true = df[LABEL_COLUMN].values

    # ---- Train / test split -------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_true, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_true
    )

    # ---- Feature scaling ----------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- Train Isolation Forest ---------------------------------------------
    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train_scaled)

    # ---- Evaluate -----------------------------------------------------------
    # Isolation Forest returns  1 → inlier,  -1 → outlier
    raw_preds = model.predict(X_test_scaled)
    y_pred = np.where(raw_preds == -1, 1, 0)  # map to 0/1

    report = classification_report(y_test, y_pred, target_names=["NORMAL", "FAULT"])
    cm = confusion_matrix(y_test, y_pred)

    print("=" * 60)
    print("CLASSIFICATION REPORT")
    print("=" * 60)
    print(report)
    print("Confusion Matrix:")
    print(cm)
    print("=" * 60)

    # ---- Persist model & scaler ---------------------------------------------
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"\nModel saved  -> {model_path}")
    print(f"Scaler saved -> {scaler_path}")

    return {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train anomaly detection model")
    parser.add_argument(
        "--dataset", default=DEFAULT_DATASET, help="Path to labelled CSV"
    )
    parser.add_argument(
        "--output", default=DEFAULT_MODEL_PATH, help="Path to save trained model"
    )
    parser.add_argument(
        "--scaler", default=DEFAULT_SCALER_PATH, help="Path to save fitted scaler"
    )
    args = parser.parse_args()
    train(dataset_path=args.dataset, model_path=args.output, scaler_path=args.scaler)


if __name__ == "__main__":
    main()
