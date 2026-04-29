"""Tests for model training pipeline."""

import os

import numpy as np
import pandas as pd
import pytest

from generate_dataset import generate_dataset
from train_model import FEATURE_COLUMNS, LABEL_COLUMN, train


class TestTrainModel:
    """End-to-end tests for the training pipeline."""

    @pytest.fixture(scope="class")
    def trained_artifacts(self, tmp_path_factory):
        """Generate dataset, train model, return paths and metrics."""
        base = tmp_path_factory.mktemp("training")
        dataset_path = os.path.join(str(base), "data.csv")
        model_path = os.path.join(str(base), "model.joblib")
        scaler_path = os.path.join(str(base), "scaler.joblib")

        # Generate a small dataset
        generate_dataset(
            total_samples=500,
            fault_ratio=0.15,
            window_size=10,
            output_path=dataset_path,
        )

        # Train
        metrics = train(
            dataset_path=dataset_path,
            model_path=model_path,
            scaler_path=scaler_path,
        )

        return {
            "dataset_path": dataset_path,
            "model_path": model_path,
            "scaler_path": scaler_path,
            "metrics": metrics,
        }

    def test_model_file_created(self, trained_artifacts):
        assert os.path.isfile(trained_artifacts["model_path"])

    def test_scaler_file_created(self, trained_artifacts):
        assert os.path.isfile(trained_artifacts["scaler_path"])

    def test_model_file_size(self, trained_artifacts):
        """Model file should be reasonably sized (< 5 MB for edge deployment)."""
        size = os.path.getsize(trained_artifacts["model_path"])
        assert size > 0, "Model file is empty"
        assert size < 5 * 1024 * 1024, f"Model file too large: {size} bytes"

    def test_metrics_returned(self, trained_artifacts):
        metrics = trained_artifacts["metrics"]
        assert "train_size" in metrics
        assert "test_size" in metrics
        assert "report" in metrics
        assert "confusion_matrix" in metrics

    def test_train_test_split_sizes(self, trained_artifacts):
        metrics = trained_artifacts["metrics"]
        total = metrics["train_size"] + metrics["test_size"]
        # 500 samples - (window_size - 1) warmup = 491
        assert total == 491

    def test_confusion_matrix_shape(self, trained_artifacts):
        cm = trained_artifacts["metrics"]["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_model_can_be_loaded(self, trained_artifacts):
        """Verify the saved model can be loaded and used for inference."""
        import joblib
        model = joblib.load(trained_artifacts["model_path"])
        scaler = joblib.load(trained_artifacts["scaler_path"])

        # Create a dummy sample
        sample = np.array([[0.15, 0.14, 0.02, 0.10, 0.20, 0.01]])
        scaled = scaler.transform(sample)
        prediction = model.predict(scaled)
        assert prediction[0] in (1, -1), "Unexpected prediction value"

    def test_model_predicts_fault_for_extreme_values(self, trained_artifacts):
        """Model should flag extreme vibration values as anomalous."""
        import joblib
        model = joblib.load(trained_artifacts["model_path"])
        scaler = joblib.load(trained_artifacts["scaler_path"])

        # Extreme sample (high vibration, high std, large delta)
        extreme = np.array([[1.8, 1.5, 0.6, 0.8, 2.0, 1.2]])
        scaled = scaler.transform(extreme)
        prediction = model.predict(scaled)
        # -1 = outlier (fault)
        assert prediction[0] == -1, "Model should detect extreme values as fault"

    def test_model_predicts_normal_for_baseline_values(self, trained_artifacts):
        """Model should not flag typical baseline values as highly anomalous."""
        import joblib
        model = joblib.load(trained_artifacts["model_path"])
        scaler = joblib.load(trained_artifacts["scaler_path"])

        # Normal baseline sample
        normal = np.array([[0.12, 0.11, 0.015, 0.08, 0.15, 0.005]])
        scaled = scaler.transform(normal)
        score = model.decision_function(scaled)[0]
        # The decision score should not be extremely negative
        # (more negative = more anomalous)
        # For a normal sample, score should be > -0.3
        assert score > -0.3, (
            f"Baseline sample flagged as too anomalous (score={score:.4f})"
        )
