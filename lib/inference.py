"""Shared inference engine for Vercel serverless functions."""

import os

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


class InferenceEngine:
    """Lightweight ONNX inference engine for anomaly detection."""

    def __init__(self):
        self._ae_session = None
        self._lstm_session = None
        self._fitted = False

    def _load_models(self):
        if not HAS_ONNX:
            return
        ae_path = os.path.join(MODEL_DIR, "autoencoder.onnx")
        lstm_path = os.path.join(MODEL_DIR, "lstm_predictor.onnx")
        if os.path.exists(ae_path):
            self._ae_session = ort.InferenceSession(ae_path)
        if os.path.exists(lstm_path):
            self._lstm_session = ort.InferenceSession(lstm_path)
        self._fitted = self._ae_session is not None

    @property
    def ready(self) -> bool:
        if not self._fitted:
            self._load_models()
        return self._fitted

    def detect(
        self, metrics: dict[str, float], history: list[list[float]] | None = None
    ) -> dict:
        values = [metrics.get(m, 0.0) for m in METRIC_NAMES]
        vector = np.array(values, dtype=np.float32).reshape(1, -1)

        ae_error = 0.0
        lstm_error = 0.0

        if self._ae_session:
            reconstructed = self._ae_session.run(None, {"input": vector})[0]
            ae_error = float(np.mean((vector - reconstructed) ** 2))

        if self._lstm_session and history and len(history) >= 30:
            seq = np.array(history[-30:], dtype=np.float32).reshape(1, 30, -1)
            predicted = self._lstm_session.run(None, {"input": seq})[0]
            lstm_error = float(np.mean((vector - predicted) ** 2))

        combined = 0.6 * ae_error + 0.4 * lstm_error
        threshold = 0.85

        if combined > threshold * 2:
            severity = "critical"
        elif combined > threshold * 1.5:
            severity = "high"
        elif combined > threshold:
            severity = "medium"
        else:
            severity = "low"

        return {
            "is_anomaly": combined > threshold,
            "severity": severity,
            "combined_score": round(combined, 6),
            "reconstruction_error": round(ae_error, 6),
            "prediction_error": round(lstm_error, 6),
            "threshold": threshold,
        }


engine = InferenceEngine()
