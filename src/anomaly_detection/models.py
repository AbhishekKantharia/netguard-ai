"""PyTorch models for network anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn


class Autoencoder(nn.Module):
    """Autoencoder for detecting anomalies in network telemetry.

    The model learns a compressed representation of normal network behavior.
    High reconstruction error indicates anomalous conditions.
    """

    def __init__(self, input_dim: int = 12, latent_dim: int = 4) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Per-sample reconstruction error (MSE)."""
        reconstructed, _ = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)


class LSTMPredictor(nn.Module):
    """LSTM-based time-series predictor for network metrics.

    Predicts next-step values; large prediction errors signal emerging anomalies.
    """

    def __init__(
        self, input_dim: int = 12, hidden_dim: int = 64, num_layers: int = 2, seq_len: int = 30
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.fc = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, input_dim)
        Returns:
            predictions: (batch, input_dim) - next-step prediction
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

    def get_prediction_error(
        self, x: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Per-sample prediction error between predicted and actual next step."""
        prediction = self.forward(x)
        return torch.mean((prediction - target) ** 2, dim=1)
