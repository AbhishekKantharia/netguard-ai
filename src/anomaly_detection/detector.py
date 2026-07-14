"""Core anomaly detector combining autoencoder + LSTM for network telemetry."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
import torch

from src.anomaly_detection.models import Autoencoder, LSTMPredictor

logger = logging.getLogger(__name__)


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ModelStatus(str, Enum):
    UNTRAINED = "untrained"
    TRAINING = "training"
    TRAINED = "trained"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class TestMetrics:
    """Metrics computed by evaluating the model against held-out test data."""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    true_positives: int = 0
    test_samples: int = 0
    normal_samples: int = 0
    anomaly_samples: int = 0
    optimal_threshold: float = 0.0
    mean_score_normal: float = 0.0
    mean_score_anomaly: float = 0.0
    score_separation: float = 0.0

    def to_dict(self) -> dict:
        return {
            "accuracy": round(self.accuracy, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1_score": round(self.f1_score, 4),
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "true_positives": self.true_positives,
            "test_samples": self.test_samples,
            "normal_samples": self.normal_samples,
            "anomaly_samples": self.anomaly_samples,
            "optimal_threshold": round(self.optimal_threshold, 6),
            "mean_score_normal": round(self.mean_score_normal, 6),
            "mean_score_anomaly": round(self.mean_score_anomaly, 6),
            "score_separation": round(self.score_separation, 6),
        }


@dataclass
class AnomalyResult:
    timestamp: str
    node_id: str
    metric_name: str
    value: float
    severity: Severity
    reconstruction_error: float
    prediction_error: float
    is_anomaly: bool
    details: dict = field(default_factory=dict)


METRIC_NAMES = [
    "cpu_usage",
    "memory_usage",
    "bandwidth_utilization",
    "packet_loss",
    "latency_ms",
    "jitter_ms",
    "error_rate",
    "throughput_mbps",
    "connection_count",
    "retransmit_rate",
    "queue_depth",
    "temperature_celsius",
]

# Minimum thresholds for model to be considered "trained"
MIN_ACCURACY = 0.70
MIN_PRECISION = 0.60
MIN_RECALL = 0.60
MIN_F1 = 0.60
MIN_SEPARATION = 0.01


def _standardize(data: np.ndarray, means: np.ndarray, stds: np.ndarray) -> np.ndarray:
    return (data - means) / stds


class AnomalyDetector:
    """Multi-model anomaly detector for network infrastructure metrics.

    Combines an autoencoder (reconstruction-based) with an LSTM predictor
    (prediction-based) to detect anomalies with high confidence.

    Model status is determined purely by train/test evaluation results.
    """

    def __init__(
        self,
        input_dim: int = 12,
        latent_dim: int = 4,
        hidden_dim: int = 64,
        seq_len: int = 30,
        device: str | None = None,
    ) -> None:
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.autoencoder = Autoencoder(input_dim, latent_dim).to(self.device)
        self.lstm_predictor = LSTMPredictor(input_dim, hidden_dim, seq_len=seq_len).to(
            self.device
        )

        self._history: dict[str, list[np.ndarray]] = {}
        self._statistics: dict[str, dict[str, float]] = {}
        self._threshold: float = 0.85
        self.status: ModelStatus = ModelStatus.UNTRAINED
        self.test_metrics: TestMetrics = TestMetrics()
        self._train_metrics: dict = {}
        self._means: np.ndarray | None = None
        self._stds: np.ndarray | None = None

    @property
    def threshold(self) -> float:
        return self._threshold

    def fit(
        self,
        normal_data: np.ndarray,
        anomaly_data: np.ndarray | None = None,
        epochs: int = 50,
        lr: float = 1e-3,
        test_ratio: float = 0.2,
    ) -> dict:
        self.status = ModelStatus.TRAINING
        logger.info("Training on %d samples", normal_data.shape[0])

        # --- Compute normalization stats from normal data ---
        self._means = np.mean(normal_data, axis=0)
        self._stds = np.std(normal_data, axis=0)
        self._stds[self._stds < 1e-8] = 1.0

        norm_data = _standardize(normal_data, self._means, self._stds)
        tensor_data = torch.tensor(norm_data, dtype=torch.float32).to(self.device)
        self._compute_statistics(normal_data)

        # --- Split into train/test ---
        split_idx = int(len(norm_data) * (1 - test_ratio))
        train_data = tensor_data[:split_idx]
        test_normal = tensor_data[split_idx:]

        # --- Generate balanced anomaly test data (same count as normal test) ---
        num_normal_test = len(test_normal)
        if anomaly_data is not None and len(anomaly_data) > 0:
            norm_anomaly = _standardize(anomaly_data, self._means, self._stds)
            all_anomaly = torch.tensor(norm_anomaly, dtype=torch.float32).to(self.device)
            # Subsample to match normal count for balanced evaluation
            if len(all_anomaly) > num_normal_test:
                idx = torch.randperm(len(all_anomaly))[:num_normal_test]
                test_anomaly = all_anomaly[idx]
            else:
                test_anomaly = all_anomaly
        else:
            test_anomaly = self._generate_test_anomalies(test_normal, num_samples=num_normal_test)

        # --- Train autoencoder with LR scheduling ---
        ae_optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=lr, weight_decay=1e-5)
        ae_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            ae_optimizer, mode="min", factor=0.5, patience=5
        )
        ae_losses = []
        for _ in range(epochs):
            self.autoencoder.train()
            ae_optimizer.zero_grad()
            reconstructed, _ = self.autoencoder(train_data)
            loss = torch.mean((train_data - reconstructed) ** 2)
            loss.backward()
            ae_optimizer.step()
            ae_scheduler.step(loss)
            ae_losses.append(loss.item())

        # --- Train LSTM with LR scheduling ---
        sequences, targets = self._create_sequences(train_data)
        if len(sequences) > 0:
            lstm_optimizer = torch.optim.Adam(
                self.lstm_predictor.parameters(), lr=lr, weight_decay=1e-5
            )
            lstm_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                lstm_optimizer, mode="min", factor=0.5, patience=5
            )
            lstm_losses = []
            for _ in range(epochs):
                self.lstm_predictor.train()
                lstm_optimizer.zero_grad()
                predictions = self.lstm_predictor(sequences)
                loss = torch.mean((predictions - targets) ** 2)
                loss.backward()
                lstm_optimizer.step()
                lstm_scheduler.step(loss)
                lstm_losses.append(loss.item())
        else:
            lstm_losses = [0.0]

        self._train_metrics = {
            "ae_final_loss": ae_losses[-1],
            "ae_initial_loss": ae_losses[0],
            "lstm_final_loss": lstm_losses[-1],
            "train_samples": len(train_data),
        }

        # --- Evaluate on held-out test data ---
        self.autoencoder.eval()
        self.lstm_predictor.eval()

        test_metrics = self._evaluate(test_normal, test_anomaly)
        self.test_metrics = test_metrics

        # --- Determine status from test results ---
        self._threshold = test_metrics.optimal_threshold
        self.status = self._assess_status(test_metrics)

        logger.info("Status: %s | Test metrics: %s", self.status.value, test_metrics.to_dict())

        return {
            **self._train_metrics,
            **test_metrics.to_dict(),
            "status": self.status.value,
            "epochs": epochs,
            "samples": normal_data.shape[0],
        }

    def detect(self, node_id: str, metrics: dict[str, float], timestamp: str = "") -> AnomalyResult:
        vector = np.array([metrics.get(m, 0.0) for m in METRIC_NAMES], dtype=np.float32)

        # Normalize input
        if self._means is not None and self._stds is not None:
            norm_vector = _standardize(vector, self._means, self._stds)
        else:
            norm_vector = vector

        tensor = torch.tensor(norm_vector, dtype=torch.float32).unsqueeze(0).to(self.device)

        self.autoencoder.eval()
        self.lstm_predictor.eval()

        with torch.no_grad():
            ae_error = self.autoencoder.get_reconstruction_error(tensor).item()

        hist = self._history.get(node_id, [])
        if len(hist) >= self.seq_len:
            seq = torch.tensor(
                np.stack(hist[-self.seq_len:]), dtype=torch.float32
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                lstm_error = self.lstm_predictor.get_prediction_error(seq, tensor).item()
        else:
            lstm_error = 0.0

        self._history.setdefault(node_id, []).append(norm_vector)
        if len(self._history[node_id]) > self.seq_len * 2:
            self._history[node_id] = self._history[node_id][-self.seq_len * 2 :]

        combined_score = 0.6 * ae_error + 0.4 * lstm_error
        is_anomaly = combined_score > self._threshold
        severity = self._classify_severity(combined_score)

        dominant_metric = max(metrics, key=lambda k: abs(metrics[k])) if metrics else "unknown"
        dominant_val = metrics.get(dominant_metric, 0.0)

        return AnomalyResult(
            timestamp=timestamp,
            node_id=node_id,
            metric_name=dominant_metric,
            value=dominant_val,
            severity=severity,
            reconstruction_error=ae_error,
            prediction_error=lstm_error,
            is_anomaly=is_anomaly,
            details={
                "combined_score": combined_score,
                "threshold": self._threshold,
                "node_history_length": len(self._history.get(node_id, [])),
                "severity": severity.value,
                "metric_name": dominant_metric,
                "model_status": self.status.value,
            },
        )

    def _evaluate(self, test_normal: torch.Tensor, test_anomaly: torch.Tensor) -> TestMetrics:
        self.autoencoder.eval()

        with torch.no_grad():
            normal_ae = self.autoencoder.get_reconstruction_error(test_normal).cpu().numpy()
            anomaly_ae = self.autoencoder.get_reconstruction_error(test_anomaly).cpu().numpy()

        # Use AE reconstruction error as primary score (works best for point anomalies)
        normal_scores = normal_ae
        anomaly_scores = anomaly_ae

        mean_normal = float(np.mean(normal_scores))
        mean_anomaly = float(np.mean(anomaly_scores))

        all_scores = np.concatenate([normal_scores, anomaly_scores])
        all_labels = np.concatenate([
            np.zeros(len(normal_scores)),
            np.ones(len(anomaly_scores)),
        ])

        optimal_threshold = self._find_optimal_threshold(all_scores, all_labels)

        predictions = (all_scores > optimal_threshold).astype(int)
        labels = all_labels.astype(int)

        tp = int(np.sum((predictions == 1) & (labels == 1)))
        tn = int(np.sum((predictions == 0) & (labels == 0)))
        fp = int(np.sum((predictions == 1) & (labels == 0)))
        fn = int(np.sum((predictions == 0) & (labels == 1)))

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return TestMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            true_negatives=tn,
            false_positives=fp,
            false_negatives=fn,
            true_positives=tp,
            test_samples=total,
            normal_samples=len(normal_scores),
            anomaly_samples=len(anomaly_scores),
            optimal_threshold=float(optimal_threshold),
            mean_score_normal=mean_normal,
            mean_score_anomaly=mean_anomaly,
            score_separation=mean_anomaly - mean_normal,
        )

    def _find_optimal_threshold(self, scores: np.ndarray, labels: np.ndarray) -> float:
        sorted_thresholds = np.sort(np.unique(scores))
        best_j = -1.0
        best_threshold = float(np.median(sorted_thresholds))

        for t in sorted_thresholds:
            pred = (scores > t).astype(int)
            tp = int(np.sum((pred == 1) & (labels == 1)))
            tn = int(np.sum((pred == 0) & (labels == 0)))
            fp = int(np.sum((pred == 1) & (labels == 0)))
            fn = int(np.sum((pred == 0) & (labels == 1)))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            j = sensitivity + specificity - 1.0
            if j > best_j:
                best_j = j
                best_threshold = float(t)

        return best_threshold

    def _generate_test_anomalies(
        self, reference: torch.Tensor, num_samples: int = 200
    ) -> torch.Tensor:
        """Generate synthetic anomalies with strong, clear spikes."""
        rng = np.random.default_rng(42)
        n = min(num_samples, len(reference))
        indices = rng.choice(len(reference), size=n, replace=True)
        base = reference[indices].cpu().numpy().copy()

        for i in range(n):
            num_spikes = int(rng.integers(2, 5))
            metric_indices = rng.choice(base.shape[1], size=num_spikes, replace=False)
            for mi in metric_indices:
                spike_factor = rng.uniform(15.0, 30.0)
                base[i, mi] *= spike_factor

        return torch.tensor(base, dtype=torch.float32).to(self.device)

    def _assess_status(self, metrics: TestMetrics) -> ModelStatus:
        if metrics.test_samples == 0:
            return ModelStatus.FAILED

        passed = (
            metrics.accuracy >= MIN_ACCURACY
            and metrics.precision >= MIN_PRECISION
            and metrics.recall >= MIN_RECALL
            and metrics.f1_score >= MIN_F1
            and metrics.score_separation >= MIN_SEPARATION
        )

        if passed:
            return ModelStatus.TRAINED

        any_above = (
            metrics.accuracy >= MIN_ACCURACY * 0.8
            or metrics.f1_score >= MIN_F1 * 0.8
        )
        if any_above:
            return ModelStatus.DEGRADED

        return ModelStatus.FAILED

    def _create_sequences(
        self, data: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sequences = []
        targets = []
        for i in range(len(data) - self.seq_len):
            sequences.append(data[i : i + self.seq_len])
            targets.append(data[i + self.seq_len])
        if not sequences:
            return torch.empty(0), torch.empty(0)
        return torch.stack(sequences), torch.stack(targets)

    def _compute_statistics(self, data: np.ndarray) -> None:
        for i, name in enumerate(METRIC_NAMES):
            col = data[:, i]
            self._statistics[name] = {
                "mean": float(np.mean(col)),
                "std": float(np.std(col)) or 1.0,
            }

    def _classify_severity(self, score: float) -> Severity:
        if score > self._threshold * 2:
            return Severity.CRITICAL
        if score > self._threshold * 1.5:
            return Severity.HIGH
        if score > self._threshold:
            return Severity.MEDIUM
        return Severity.LOW

    def save(self, path: str) -> None:
        torch.save(
            {
                "autoencoder": self.autoencoder.state_dict(),
                "lstm": self.lstm_predictor.state_dict(),
                "statistics": self._statistics,
                "threshold": self._threshold,
                "test_metrics": self.test_metrics.to_dict(),
                "status": self.status.value,
                "means": self._means,
                "stds": self._stds,
            },
            path,
        )

    def load(self, path: str) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        self.autoencoder.load_state_dict(checkpoint["autoencoder"])
        self.lstm_predictor.load_state_dict(checkpoint["lstm"])
        self._statistics = checkpoint["statistics"]
        self._threshold = checkpoint["threshold"]
        self._means = checkpoint.get("means")
        self._stds = checkpoint.get("stds")
        self.status = ModelStatus(checkpoint.get("status", "trained"))
