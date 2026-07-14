"""Synthetic network telemetry data generator for training and demos."""

from __future__ import annotations

import numpy as np

from src.anomaly_detection.detector import METRIC_NAMES


def generate_normal_data(
    num_samples: int = 10000, num_metrics: int = 12, seed: int = 42
) -> np.ndarray:
    """Generate synthetic 'normal' network telemetry data.

    Each metric has realistic baseline distributions:
    - CPU: 20-60%
    - Memory: 40-75%
    - Bandwidth: 10-80%
    - Packet loss: 0-2%
    - Latency: 5-50ms
    - Jitter: 1-10ms
    - Error rate: 0-1%
    - Throughput: 50-500 Mbps
    - Connections: 100-5000
    - Retransmit rate: 0-0.5%
    - Queue depth: 0-100
    - Temperature: 35-65C
    """
    rng = np.random.default_rng(seed)

    baselines = {
        "cpu_usage": (40, 10),
        "memory_usage": (58, 8),
        "bandwidth_utilization": (45, 15),
        "packet_loss": (0.5, 0.3),
        "latency_ms": (20, 8),
        "jitter_ms": (4, 2),
        "error_rate": (0.2, 0.1),
        "throughput_mbps": (250, 80),
        "connection_count": (2000, 800),
        "retransmit_rate": (0.1, 0.08),
        "queue_depth": (30, 20),
        "temperature_celsius": (48, 6),
    }

    data = np.zeros((num_samples, num_metrics), dtype=np.float32)
    for i, name in enumerate(METRIC_NAMES[:num_metrics]):
        mean, std = baselines.get(name, (50, 10))
        data[:, i] = rng.normal(mean, std, num_samples)

    data = np.clip(data, 0, None)
    return data


def inject_anomalies(
    data: np.ndarray, num_anomalies: int = 100, severity: float = 4.0, seed: int = 99
) -> tuple[np.ndarray, list[dict]]:
    """Inject anomalies into normal data by spiking multiple metrics simultaneously.

    Anomalies are created by:
    1. Spiking 2-4 metrics at once with 5-15x amplification
    2. Adding correlated noise across metrics
    3. Creating extreme outliers in critical metrics

    Returns:
        Modified data array and list of anomaly metadata dicts.
    """
    rng = np.random.default_rng(seed)
    anomalous = data.copy()
    metadata = []

    indices = rng.choice(len(data), size=min(num_anomalies, len(data)), replace=False)
    for idx in indices:
        num_spikes = int(rng.integers(2, 5))
        metric_indices = rng.choice(data.shape[1], size=num_spikes, replace=False)
        spike_info = []

        for metric_idx in metric_indices:
            spike = severity * rng.uniform(4.0, 12.0)
            original = anomalous[idx, metric_idx]
            anomalous[idx, metric_idx] = original * spike
            spike_info.append({
                "metric": METRIC_NAMES[metric_idx],
                "original_value": float(original),
                "anomalous_value": float(anomalous[idx, metric_idx]),
            })

        metadata.append({
            "index": int(idx),
            "spikes": spike_info,
            "severity": severity,
        })

    return anomalous, metadata


def generate_time_series(
    num_nodes: int = 5, num_timesteps: int = 500, seed: int = 42
) -> dict[str, np.ndarray]:
    """Generate time-series data for multiple network nodes.

    Returns:
        Dict mapping node_id to array of shape (timesteps, metrics).
    """
    rng = np.random.default_rng(seed)
    series = {}
    for node_idx in range(num_nodes):
        node_id = f"node-{node_idx:03d}"
        base = generate_normal_data(num_timesteps, seed=int(rng.integers(0, 10000)))
        scale = rng.uniform(0.5, 2.0, (1, base.shape[1]))
        trend = np.linspace(0, 5, num_timesteps).reshape(-1, 1) * scale
        noise = rng.normal(0, 1, base.shape)
        series[node_id] = base + trend + noise
    return series
