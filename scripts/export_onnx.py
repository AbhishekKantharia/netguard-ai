"""Export trained PyTorch models to ONNX for Vercel deployment."""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.anomaly_detection.detector import AnomalyDetector
from src.anomaly_detection.data_generator import generate_normal_data, inject_anomalies
from src.anomaly_detection.models import Autoencoder, LSTMPredictor

ONNX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "models")
os.makedirs(ONNX_DIR, exist_ok=True)


def train_and_export():
    print("Training models...")
    detector = AnomalyDetector()

    normal_data = generate_normal_data(600)
    full_anomaly, metadata = inject_anomalies(normal_data, num_anomalies=150, severity=6.0)
    anomaly_indices = [m["index"] for m in metadata]
    anomaly_only = full_anomaly[anomaly_indices]

    result = detector.fit(normal_data, anomaly_only, epochs=30, lr=5e-4)
    print("Training complete: status=%s, accuracy=%s, f1=%s" % (result['status'], result['accuracy'], result['f1_score']))

    # Export autoencoder
    ae_path = os.path.join(ONNX_DIR, "autoencoder.onnx")
    detector.autoencoder.eval()
    dummy_ae = torch.randn(1, 12)
    torch.onnx.export(
        detector.autoencoder, dummy_ae, ae_path,
        input_names=["input"],
        output_names=["reconstructed", "latent"],
        opset_version=18,
    )
    print("Exported autoencoder to %s" % ae_path)

    # Export LSTM
    lstm_path = os.path.join(ONNX_DIR, "lstm_predictor.onnx")
    detector.lstm_predictor.eval()
    dummy_lstm = torch.randn(1, 30, 12)
    torch.onnx.export(
        detector.lstm_predictor, dummy_lstm, lstm_path,
        input_names=["input"],
        output_names=["prediction"],
        opset_version=18,
    )
    print("Exported LSTM to %s" % lstm_path)

    # Export normalization stats + threshold as numpy
    stats_path = os.path.join(ONNX_DIR, "norm_stats.npz")
    np.savez(
        stats_path,
        means=detector._means,
        stds=detector._stds,
        threshold=np.array([detector._threshold]),
    )
    print("Exported normalization stats to %s" % stats_path)
    print("Threshold: %s" % detector._threshold)


if __name__ == "__main__":
    train_and_export()
    print("ONNX export complete")
