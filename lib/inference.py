"""Shared inference engine for Vercel serverless functions."""

import os
import json

import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False

METRIC_NAMES = [
    "cpu_usage", "memory_usage", "bandwidth_utilization", "packet_loss",
    "latency_ms", "jitter_ms", "error_rate", "throughput_mbps",
    "connection_count", "retransmit_rate", "queue_depth", "temperature_celsius",
]

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "public", "models")
ALT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
API_MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "api", "models")


class InferenceEngine:
    """Lightweight ONNX inference engine for anomaly detection."""

    def __init__(self):
        self._ae_session = None
        self._lstm_session = None
        self._fitted = False
        self._means = None
        self._stds = None
        self._threshold = 0.85
        self._history = {}

    def _load_models(self):
        if not HAS_ONNX:
            return
        for base in [MODEL_DIR, ALT_MODEL_DIR, API_MODEL_DIR, "/var/task/public/models"]:
            ae_path = os.path.join(base, "autoencoder.onnx")
            lstm_path = os.path.join(base, "lstm_predictor.onnx")
            stats_path = os.path.join(base, "norm_stats.npz")
            if os.path.exists(ae_path):
                self._ae_session = ort.InferenceSession(ae_path)
                if os.path.exists(lstm_path):
                    self._lstm_session = ort.InferenceSession(lstm_path)
                if os.path.exists(stats_path):
                    stats = np.load(stats_path)
                    self._means = stats["means"]
                    self._stds = stats["stds"]
                    self._threshold = float(stats["threshold"][0])
                self._fitted = True
                return
        self._fitted = self._ae_session is not None

    @property
    def ready(self) -> bool:
        if not self._fitted:
            self._load_models()
        return self._fitted

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        if self._means is not None and self._stds is not None:
            return (vector - self._means) / self._stds
        return vector

    _DEFAULTS = {
        "cpu_usage": 40.0,
        "memory_usage": 58.0,
        "bandwidth_utilization": 45.0,
        "packet_loss": 0.5,
        "latency_ms": 20.0,
        "jitter_ms": 4.0,
        "error_rate": 0.2,
        "throughput_mbps": 250.0,
        "connection_count": 2000.0,
        "retransmit_rate": 0.15,
        "queue_depth": 50.0,
        "temperature_celsius": 50.0,
    }

    def detect(
        self, metrics: dict[str, float], history: list[list[float]] | None = None
    ) -> dict:
        self.ready
        values = [metrics.get(m, self._DEFAULTS.get(m, 0.0)) for m in METRIC_NAMES]
        raw_vector = np.array(values, dtype=np.float32).reshape(1, -1)
        norm_vector = self._normalize(raw_vector)

        ae_error = 0.0
        lstm_error = 0.0

        if self._ae_session:
            reconstructed = self._ae_session.run(None, {"input": norm_vector})[0]
            ae_error = float(np.mean((norm_vector - reconstructed) ** 2))

        node_id = metrics.get("_node_id", "default")
        hist = self._history.get(node_id, [])
        hist.append(norm_vector.flatten().tolist())
        if len(hist) > 30:
            hist = hist[-30:]
        self._history[node_id] = hist

        if self._lstm_session and len(hist) >= 30:
            seq = np.array(hist[-30:], dtype=np.float32).reshape(1, 30, -1)
            predicted = self._lstm_session.run(None, {"input": seq})[0]
            lstm_error = float(np.mean((norm_vector - predicted) ** 2))

        combined = 0.6 * ae_error + 0.4 * lstm_error

        if combined > self._threshold * 2:
            severity = "critical"
        elif combined > self._threshold * 1.5:
            severity = "high"
        elif combined > self._threshold:
            severity = "medium"
        else:
            severity = "low"

        return {
            "is_anomaly": combined > self._threshold,
            "severity": severity,
            "combined_score": round(combined, 6),
            "reconstruction_error": round(ae_error, 6),
            "prediction_error": round(lstm_error, 6),
            "threshold": round(self._threshold, 6),
            "model_status": "trained" if self._fitted else "untrained",
        }


engine = InferenceEngine()
