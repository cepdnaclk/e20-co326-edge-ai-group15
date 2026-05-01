"""Train a One-Class SVM anomaly-detection model and persist it to disk.

Model  : One-Class SVM (RBF kernel)
Task   : Unsupervised anomaly detection on vibration sensor data
Output : anomaly_detector.joblib  — trained One-Class SVM
         scaler.joblib            — fitted StandardScaler
         threshold.joblib         — optimised decision threshold (float)

Data splits
-----------
  80 %  train_full  ┐
    ├─ 80 % train_gs  ┐  used for grid-search (scaler fit + SVM fit)
    │    └─ 85 % train   → SVM trained on NORMAL samples only
    │    └─ 15 % thr_val → threshold tuning for final model
    └─ 20 % val          → hyperparameter selection during grid search
  20 %  test             → final held-out evaluation (never touched earlier)

Usage
-----
    python train_model.py
    python train_model.py --dataset dataset/vibration_data.csv \\
                          --output  model/anomaly_detector.joblib

Inference
---------
    model     = joblib.load("model/anomaly_detector.joblib")
    scaler    = joblib.load("model/scaler.joblib")
    threshold = joblib.load("model/threshold.joblib")

    scores = -model.decision_function(scaler.transform(X_new))
    y_pred = (scores >= threshold).astype(int)   # 1 = FAULT, 0 = NORMAL
"""

import argparse
import os
import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR            = os.path.dirname(__file__)
DEFAULT_DATASET     = os.path.join(BASE_DIR, "dataset", "vibration_data.csv")
DEFAULT_MODEL_DIR   = os.path.join(BASE_DIR, "model")
DEFAULT_MODEL_PATH  = os.path.join(DEFAULT_MODEL_DIR, "anomaly_detector.joblib")
DEFAULT_SCALER_PATH = os.path.join(DEFAULT_MODEL_DIR, "scaler.joblib")
DEFAULT_THR_PATH    = os.path.join(DEFAULT_MODEL_DIR, "threshold.joblib")

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
FEATURE_COLUMNS = [
    "vibration",
    "rolling_mean",
    "rolling_std",
    "rolling_min",
    "rolling_max",
    "delta",
]
LABEL_COLUMN = "label"   # 0 = NORMAL, 1 = FAULT

# ---------------------------------------------------------------------------
# Hyperparameter grid
# ---------------------------------------------------------------------------
PARAM_GRID = {
    "nu":    [0.05, 0.10, 0.15, 0.20],
    "gamma": ["scale", "auto", 0.01, 0.05, 0.1],
}

RANDOM_STATE = 42
TEST_SIZE    = 0.20


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_best_threshold(y: np.ndarray, scores: np.ndarray, n_steps: int = 300) -> float:
    """Return the anomaly-score threshold that maximises F1 on (y, scores).

    Args:
        y:       True binary labels (0 = NORMAL, 1 = FAULT).
        scores:  Anomaly scores — higher means more anomalous.
        n_steps: Number of candidate thresholds to sweep.

    Returns:
        Threshold value (float) that maximises F1.
    """
    thresholds = np.linspace(scores.min(), scores.max(), n_steps)
    best_f1, best_thr = 0.0, thresholds[0]
    for thr in thresholds:
        f1 = f1_score(y, (scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    dataset_path: str = DEFAULT_DATASET,
    model_path:   str = DEFAULT_MODEL_PATH,
    scaler_path:  str = DEFAULT_SCALER_PATH,
    thr_path:     str = DEFAULT_THR_PATH,
) -> dict:
    """Train a One-Class SVM with grid-search tuning and persist artefacts.

    Args:
        dataset_path: Path to the labelled CSV file.
        model_path:   Destination path for the trained model (.joblib).
        scaler_path:  Destination path for the fitted scaler (.joblib).
        thr_path:     Destination path for the decision threshold (.joblib).

    Returns:
        Dictionary with train/test sizes and evaluation metrics.
    """

    # ------------------------------------------------------------------ data
    df = pd.read_csv(dataset_path)
    print(f"Loaded dataset : {len(df)} rows, {len(df.columns)} columns")
    print(f"Label distribution:\n{df[LABEL_COLUMN].value_counts().to_string()}\n")

    X = df[FEATURE_COLUMNS].values
    y = df[LABEL_COLUMN].values

    # ---------------------------------------------------------- data splits
    # 1. Hold out 20 % as the final test set (never touched during tuning)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 2. From train_full: 20 % → val (hyperparameter selection)
    X_train_gs, X_val, y_train_gs, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y_train_full,
    )

    # 3. From train_gs: 15 % → thr_val (threshold tuning for final model only)
    X_train, X_thr_val, y_train, y_thr_val = train_test_split(
        X_train_gs, y_train_gs,
        test_size=0.15,
        random_state=RANDOM_STATE + 1,
        stratify=y_train_gs,
    )

    # ------------------------------------------------------- feature scaling
    # Scaler fitted ONLY on X_train to prevent leakage into val / test.
    scaler_gs      = StandardScaler()
    X_train_scaled = scaler_gs.fit_transform(X_train)
    X_val_scaled   = scaler_gs.transform(X_val)

    # One-Class SVM trains exclusively on NORMAL samples.
    X_train_normal = X_train_scaled[y_train == 0]
    print(f"Grid-search training : {len(X_train_normal)} normal samples "
          f"(of {len(X_train_scaled)} total)\n")

    # ----------------------------------------------------- hyperparameter search
    total = len(PARAM_GRID["nu"]) * len(PARAM_GRID["gamma"])
    print(f"Hyperparameter grid search — {total} combinations ...\n")

    best_val_f1 = -1.0
    best_params = {}

    for nu in PARAM_GRID["nu"]:
        for gamma in PARAM_GRID["gamma"]:
            m = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
            m.fit(X_train_normal)

            val_scores = -m.decision_function(X_val_scaled)
            thr        = find_best_threshold(y_val, val_scores)
            val_preds  = (val_scores >= thr).astype(int)

            f1  = f1_score(y_val, val_preds, zero_division=0)
            pre = precision_score(y_val, val_preds, zero_division=0)
            rec = recall_score(y_val, val_preds, zero_division=0)

            print(f"  nu={nu:<5}  gamma={str(gamma):<8}  "
                  f"val_F1={f1:.4f}  P={pre:.4f}  R={rec:.4f}")

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_params = {"nu": nu, "gamma": gamma}

    print(f"\nBest params  : {best_params}")
    print(f"Best val F1  : {best_val_f1:.4f}\n")

    # ---------------------------------------- final model on larger dataset
    # Refit scaler on train_gs (train + thr_val combined) for richer statistics.
    scaler            = StandardScaler()
    X_train_gs_scaled = scaler.fit_transform(X_train_gs)
    X_thr_val_scaled  = scaler.transform(X_thr_val)
    X_test_scaled     = scaler.transform(X_test)

    # Refit model on all normal samples in train_gs.
    X_train_gs_normal = X_train_gs_scaled[y_train_gs == 0]
    print(f"Final model training : {len(X_train_gs_normal)} normal samples\n")

    model = OneClassSVM(kernel="rbf", **best_params)
    model.fit(X_train_gs_normal)

    # Tune threshold on the dedicated thr_val split (unseen during grid search).
    thr_val_scores  = -model.decision_function(X_thr_val_scaled)
    final_threshold = find_best_threshold(y_thr_val, thr_val_scores)

    # ----------------------------------------------------------- evaluation
    test_scores = -model.decision_function(X_test_scaled)
    y_pred      = (test_scores >= final_threshold).astype(int)

    report   = classification_report(y_test, y_pred, target_names=["NORMAL", "FAULT"])
    cm       = confusion_matrix(y_test, y_pred)
    f1_test  = f1_score(y_test, y_pred, zero_division=0)
    pre_test = precision_score(y_test, y_pred, zero_division=0)
    rec_test = recall_score(y_test, y_pred, zero_division=0)
    auc_test = roc_auc_score(y_test, test_scores)

    print("=" * 60)
    print("FINAL TEST SET EVALUATION")
    print("=" * 60)
    print(report)
    print(f"Confusion Matrix:\n{cm}\n")
    print(f"F1        : {f1_test:.4f}")
    print(f"Precision : {pre_test:.4f}")
    print(f"Recall    : {rec_test:.4f}")
    print(f"AUC       : {auc_test:.4f}")
    print(f"Threshold : {final_threshold:.6f}")
    print("=" * 60)

    # ---------------------------------------------------- persist artefacts
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model,           model_path)
    joblib.dump(scaler,          scaler_path)
    joblib.dump(final_threshold, thr_path)

    print(f"\nModel saved     -> {model_path}")
    print(f"Scaler saved    -> {scaler_path}")
    print(f"Threshold saved -> {thr_path}")

    return {
        "train_size":       len(X_train_full),
        "test_size":        len(X_test),
        "best_params":      best_params,
        "final_threshold":  final_threshold,
        "report":           report,
        "confusion_matrix": cm.tolist(),
        "f1":               f1_test,
        "precision":        pre_test,
        "recall":           rec_test,
        "auc":              auc_test,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a One-Class SVM anomaly detection model on vibration data"
    )
    parser.add_argument("--dataset",   default=DEFAULT_DATASET,    help="Path to labelled CSV")
    parser.add_argument("--output",    default=DEFAULT_MODEL_PATH,  help="Path to save trained model")
    parser.add_argument("--scaler",    default=DEFAULT_SCALER_PATH, help="Path to save fitted scaler")
    parser.add_argument("--threshold", default=DEFAULT_THR_PATH,    help="Path to save decision threshold")
    args = parser.parse_args()
    train(
        dataset_path=args.dataset,
        model_path=args.output,
        scaler_path=args.scaler,
        thr_path=args.threshold,
    )


if __name__ == "__main__":
    main()