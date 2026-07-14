"""Tests for anomaly detection module."""

import pytest

from src.anomaly_detection.data_generator import (
    generate_normal_data,
    generate_time_series,
    inject_anomalies,
)
from src.anomaly_detection.detector import (
    METRIC_NAMES,
    AnomalyDetector,
    ModelStatus,
    Severity,
)


@pytest.fixture
def detector():
    return AnomalyDetector()


@pytest.fixture
def normal_data():
    return generate_normal_data(num_samples=500, seed=42)


@pytest.fixture
def anomaly_data():
    normal = generate_normal_data(num_samples=100, seed=99)
    _, meta = inject_anomalies(normal, num_anomalies=100)
    return normal


class TestAnomalyDetector:
    def test_init_untrained(self, detector):
        assert detector.status == ModelStatus.UNTRAINED

    def test_fit_sets_status_based_on_evaluation(self, detector, normal_data):
        result = detector.fit(normal_data, epochs=20, test_ratio=0.2)
        assert detector.status in (ModelStatus.TRAINED, ModelStatus.DEGRADED, ModelStatus.FAILED)
        assert "accuracy" in result
        assert "precision" in result
        assert "recall" in result
        assert "f1_score" in result
        assert "status" in result

    def test_test_metrics_populated_after_training(self, detector, normal_data):
        detector.fit(normal_data, epochs=20, test_ratio=0.2)
        tm = detector.test_metrics
        assert tm.test_samples > 0
        assert tm.normal_samples > 0
        assert tm.anomaly_samples > 0
        assert 0.0 <= tm.accuracy <= 1.0
        assert 0.0 <= tm.precision <= 1.0
        assert 0.0 <= tm.recall <= 1.0

    def test_threshold_determined_by_data(self, detector, normal_data):
        detector.fit(normal_data, epochs=20, test_ratio=0.2)
        assert detector.threshold > 0.0
        assert isinstance(detector.threshold, float)

    def test_detect_returns_model_status(self, detector, normal_data):
        detector.fit(normal_data, epochs=20, test_ratio=0.2)
        metrics = {name: 40.0 for name in METRIC_NAMES}
        result = detector.detect("node-001", metrics, timestamp="2026-01-01T00:00:00Z")
        assert result.node_id == "node-001"
        assert result.severity in Severity
        assert isinstance(result.is_anomaly, bool)
        assert "model_status" in result.details

    def test_detect_extreme_metrics_high_score(self, detector, normal_data):
        detector.fit(normal_data, epochs=20, test_ratio=0.2)
        metrics = {name: 500.0 for name in METRIC_NAMES}
        result = detector.detect("node-extreme", metrics)
        normal_metrics = {name: 40.0 for name in METRIC_NAMES}
        normal_result = detector.detect("node-normal-check", normal_metrics)
        assert result.details["combined_score"] > normal_result.details["combined_score"]


class TestDataGenerator:
    def test_generate_normal_data_shape(self):
        data = generate_normal_data(num_samples=100)
        assert data.shape == (100, 12)

    def test_inject_anomalies(self):
        data = generate_normal_data(num_samples=200)
        anomalous, meta = inject_anomalies(data, num_anomalies=10)
        assert len(meta) == 10
        assert anomalous.shape == data.shape

    def test_generate_time_series(self):
        series = generate_time_series(num_nodes=3, num_timesteps=100)
        assert len(series) == 3
        for node_id, arr in series.items():
            assert node_id.startswith("node-")
            assert arr.shape == (100, 12)
