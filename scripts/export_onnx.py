"""Export PyTorch models to ONNX for Vercel deployment."""

import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.anomaly_detection.models import Autoencoder, LSTMPredictor

ONNX_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "models")
os.makedirs(ONNX_DIR, exist_ok=True)


def export_autoencoder(input_dim: int = 12, latent_dim: int = 4):
    model = Autoencoder(input_dim, latent_dim)
    model.eval()
    dummy = torch.randn(1, input_dim)
    path = os.path.join(ONNX_DIR, "autoencoder.onnx")
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["reconstructed", "latent"],
        opset_version=14,
    )
    print(f"Exported autoencoder to {path}")


def export_lstm(input_dim: int = 12, hidden_dim: int = 64, seq_len: int = 30):
    model = LSTMPredictor(input_dim, hidden_dim, seq_len=seq_len)
    model.eval()
    dummy = torch.randn(1, seq_len, input_dim)
    path = os.path.join(ONNX_DIR, "lstm_predictor.onnx")
    torch.onnx.export(
        model, dummy, path,
        input_names=["input"],
        output_names=["prediction"],
        opset_version=14,
    )
    print(f"Exported LSTM predictor to {path}")


if __name__ == "__main__":
    export_autoencoder()
    export_lstm()
    print("ONNX export complete")
