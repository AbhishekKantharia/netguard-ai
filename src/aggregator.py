"""Aggregator that wires anomaly detection + self-healing into a single pipeline."""

from __future__ import annotations

import time

from src.anomaly_detection.data_generator import generate_normal_data
from src.anomaly_detection.detector import AnomalyDetector, AnomalyResult
from src.self_healing.agent import SelfHealingAgent


class NetworkAggregator:
    """Central coordinator for detection and healing."""

    def __init__(self) -> None:
        self.detector = AnomalyDetector()
        self.healer = SelfHealingAgent(enabled=True)
        self._node_data: dict[str, list[dict]] = {}
        self._anomaly_counts: dict[str, int] = {}
        self._healing_counts: dict[str, int] = {}
        self._start_time = time.time()

    def process_metrics(
        self, node_id: str, metrics: dict[str, float], timestamp: str = ""
    ) -> dict:
        """Full pipeline: detect anomaly -> heal if needed -> return result."""
        result: AnomalyResult = self.detector.detect(node_id, metrics, timestamp)

        self._node_data.setdefault(node_id, []).append(
            {"metrics": metrics, "is_anomaly": result.is_anomaly, "timestamp": timestamp}
        )
        if len(self._node_data[node_id]) > 500:
            self._node_data[node_id] = self._node_data[node_id][-500:]

        healing_event = None
        if result.is_anomaly:
            self._anomaly_counts[node_id] = self._anomaly_counts.get(node_id, 0) + 1

            import asyncio
            import concurrent.futures

            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, self.healer.heal(result))
                    event = future.result()
            except RuntimeError:
                event = asyncio.run(self.healer.heal(result))

            self._healing_counts[node_id] = self._healing_counts.get(node_id, 0) + 1
            healing_event = {
                "event_id": event.event_id,
                "strategy": event.action.strategy_name if event.action else "none",
                "status": event.action.status.value if event.action else "skipped",
                "resolution": event.resolution,
                "success": event.success,
            }

        return {
            "node_id": result.node_id,
            "metric_name": result.metric_name,
            "value": result.value,
            "severity": result.severity.value,
            "reconstruction_error": result.reconstruction_error,
            "prediction_error": result.prediction_error,
            "is_anomaly": result.is_anomaly,
            "combined_score": result.details.get("combined_score", 0.0),
            "model_status": result.details.get("model_status", "untrained"),
            "healing_event": healing_event,
        }

    def train(
        self,
        epochs: int = 30,
        num_samples: int = 600,
        test_ratio: float = 0.2,
    ) -> dict:
        normal_data = generate_normal_data(num_samples)
        return self.detector.fit(
            normal_data, epochs=epochs, test_ratio=test_ratio
        )

    def get_node_statuses(self) -> list[dict]:
        statuses = []
        for node_id, data in self._node_data.items():
            last = data[-1] if data else {}
            statuses.append(
                {
                    "node_id": node_id,
                    "status": "anomaly" if last.get("is_anomaly") else "healthy",
                    "last_check": last.get("timestamp", ""),
                    "anomaly_count_24h": self._anomaly_counts.get(node_id, 0),
                    "healing_events_24h": self._healing_counts.get(node_id, 0),
                }
            )
        return statuses

    def get_system_stats(self) -> dict:
        healing_stats = self.healer.get_stats()
        return {
            "total_nodes": len(self._node_data),
            "active_anomalies": sum(
                1
                for node_data in self._node_data.values()
                if node_data and node_data[-1].get("is_anomaly")
            ),
            "total_healing_events": healing_stats["total_events"],
            "healing_success_rate": healing_stats["success_rate"],
            "model_status": self.detector.status.value,
            "uptime_seconds": time.time() - self._start_time,
        }

    def get_healing_events(self, limit: int = 50) -> list[dict]:
        events = self.healer.get_event_log(limit)
        return [
            {
                "event_id": e.event_id,
                "timestamp": e.timestamp,
                "node_id": e.anomaly.node_id if e.anomaly else "",
                "strategy": e.action.strategy_name if e.action else "none",
                "status": e.action.status.value if e.action else "skipped",
                "resolution": e.resolution,
                "success": e.success,
            }
            for e in events
        ]
