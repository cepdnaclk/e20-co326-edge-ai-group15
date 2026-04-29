"""Tests for the dataset generation pipeline."""

import csv
import os
import tempfile

import pytest

from generate_dataset import (
    FIELDNAMES,
    compute_features,
    compute_rolling_mean,
    compute_rolling_std,
    generate_dataset,
)
from collections import deque


# ---------------------------------------------------------------------------
# Unit tests – rolling statistics helpers
# ---------------------------------------------------------------------------


class TestRollingMean:
    """Tests for compute_rolling_mean."""

    def test_single_value(self):
        window = deque([5.0])
        assert compute_rolling_mean(window) == 5.0

    def test_multiple_values(self):
        window = deque([1.0, 2.0, 3.0, 4.0, 5.0])
        assert compute_rolling_mean(window) == pytest.approx(3.0)

    def test_identical_values(self):
        window = deque([7.0, 7.0, 7.0])
        assert compute_rolling_mean(window) == pytest.approx(7.0)


class TestRollingStd:
    """Tests for compute_rolling_std."""

    def test_single_value_returns_zero(self):
        window = deque([5.0])
        assert compute_rolling_std(window, 5.0) == 0.0

    def test_identical_values_returns_zero(self):
        window = deque([3.0, 3.0, 3.0])
        assert compute_rolling_std(window, 3.0) == pytest.approx(0.0)

    def test_known_std(self):
        window = deque([2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0])
        mean = compute_rolling_mean(window)
        std = compute_rolling_std(window, mean)
        assert std > 0
        assert isinstance(std, float)


# ---------------------------------------------------------------------------
# Unit tests – feature computation
# ---------------------------------------------------------------------------


class TestComputeFeatures:
    """Tests for compute_features."""

    def test_output_keys(self):
        window = deque([0.1, 0.2, 0.3, 0.15, 0.25], maxlen=5)
        result = compute_features(window, current=0.25, previous=0.15)
        expected_keys = {"vibration", "rolling_mean", "rolling_std",
                         "rolling_min", "rolling_max", "delta"}
        assert set(result.keys()) == expected_keys

    def test_delta_calculation(self):
        window = deque([0.1, 0.2, 0.5], maxlen=3)
        result = compute_features(window, current=0.5, previous=0.2)
        assert result["delta"] == pytest.approx(0.3, abs=0.001)

    def test_rolling_min_max(self):
        window = deque([0.1, 0.5, 0.3], maxlen=3)
        result = compute_features(window, current=0.3, previous=0.5)
        assert result["rolling_min"] == pytest.approx(0.1, abs=0.001)
        assert result["rolling_max"] == pytest.approx(0.5, abs=0.001)

    def test_values_are_rounded(self):
        window = deque([0.123456789, 0.987654321], maxlen=2)
        result = compute_features(window, current=0.987654321, previous=0.123456789)
        for key, value in result.items():
            # All values should have at most 4 decimal places
            assert round(value, 4) == value, f"{key} not rounded to 4 dp"


# ---------------------------------------------------------------------------
# Integration test – full dataset generation
# ---------------------------------------------------------------------------


class TestGenerateDataset:
    """Integration tests for generate_dataset."""

    @pytest.fixture()
    def dataset_path(self, tmp_path):
        """Generate a small dataset in a temp directory and return its path."""
        path = os.path.join(str(tmp_path), "test_data.csv")
        generate_dataset(
            total_samples=200,
            fault_ratio=0.15,
            window_size=10,
            output_path=path,
        )
        return path

    def test_file_is_created(self, dataset_path):
        assert os.path.isfile(dataset_path)

    def test_correct_headers(self, dataset_path):
        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            assert set(reader.fieldnames) == set(FIELDNAMES)

    def test_row_count(self, dataset_path):
        """Rows should be total_samples minus warmup samples."""
        with open(dataset_path) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            rows = list(reader)
        # 200 samples, window_size 10: first 9 are warmup (window not full)
        # so we get total_samples - (window_size - 1) rows
        expected = 200 - (10 - 1)
        assert len(rows) == expected, f"Expected {expected} rows, got {len(rows)}"

    def test_labels_are_binary(self, dataset_path):
        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert row["label"] in ("0", "1"), f"Unexpected label: {row['label']}"

    def test_has_both_classes(self, dataset_path):
        labels = set()
        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                labels.add(row["label"])
        # With 15% fault rate over 190 rows, extremely unlikely to miss a class
        assert "0" in labels, "No normal samples found"
        assert "1" in labels, "No fault samples found"

    def test_vibration_values_are_positive(self, dataset_path):
        with open(dataset_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                assert float(row["vibration"]) >= 0.0
