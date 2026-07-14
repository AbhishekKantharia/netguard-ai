"""Self-healing agent that orchestrates remediation strategies."""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src.anomaly_detection.detector import AnomalyResult
from src.self_healing.strategies import (
    ALL_STRATEGIES,
    RemediationAction,
    RemediationStatus,
    RemediationStrategy,
)

logger = logging.getLogger(__name__)


@dataclass
class HealingEvent:
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = ""
    anomaly: AnomalyResult | None = None
    action: RemediationAction | None = None
    resolution: str = ""
    success: bool = False


class SelfHealingAgent:
    """Orchestrates automatic remediation when anomalies are detected.

    Selects the appropriate remediation strategy based on anomaly type and
    severity, executes the action, and logs the outcome.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.strategies: list[RemediationStrategy] = list(ALL_STRATEGIES)
        self._event_log: list[HealingEvent] = []
        self._active_actions: dict[str, RemediationAction] = {}

    def select_strategy(self, anomaly: AnomalyResult) -> RemediationStrategy | None:
        """Pick the best strategy for the given anomaly."""
        candidates = [s for s in self.strategies if s.can_handle(anomaly.details)]
        if not candidates:
            logger.info("No strategy matches anomaly on node %s", anomaly.node_id)
            return None

        severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        candidates.sort(key=lambda s: severity_order.get(anomaly.severity.value, 99))
        return candidates[0]

    async def heal(self, anomaly: AnomalyResult) -> HealingEvent:
        """Attempt to heal the network based on detected anomaly.

        Returns a HealingEvent describing what action was taken.
        """
        event = HealingEvent(
            timestamp=datetime.now(timezone.utc).isoformat(),
            anomaly=anomaly,
        )

        if not self.enabled:
            event.resolution = "Healing disabled; logging only"
            self._event_log.append(event)
            return event

        strategy = self.select_strategy(anomaly)
        if strategy is None:
            event.resolution = "No applicable remediation strategy"
            self._event_log.append(event)
            return event

        action = RemediationAction(
            strategy_name=strategy.name,
            target_node=anomaly.node_id,
            parameters=self._derive_parameters(anomaly),
        )

        try:
            action = await strategy.execute(action)
            event.action = action
            event.success = action.status == RemediationStatus.SUCCESS
            event.resolution = action.result
        except Exception as exc:
            action.status = RemediationStatus.FAILED
            action.result = str(exc)
            event.action = action
            event.success = False
            event.resolution = f"Healing failed: {exc}"
            logger.error("Healing failed for node %s: %s", anomaly.node_id, exc)

        self._event_log.append(event)
        if event.success:
            logger.info(
                "Healed node %s with %s: %s",
                anomaly.node_id,
                strategy.name,
                event.resolution,
            )
        return event

    def get_event_log(self, limit: int = 50) -> list[HealingEvent]:
        return self._event_log[-limit:]

    def get_stats(self) -> dict:
        total = len(self._event_log)
        successes = sum(1 for e in self._event_log if e.success)
        return {
            "total_events": total,
            "successful_heals": successes,
            "success_rate": successes / total if total > 0 else 0.0,
            "strategies_available": [s.name for s in self.strategies],
        }

    def _derive_parameters(self, anomaly: AnomalyResult) -> dict:
        """Infer remediation parameters from anomaly details."""
        params: dict = {}
        metric = anomaly.metric_name

        if metric in ("cpu_usage", "memory_usage"):
            params["resource"] = "cpu" if metric == "cpu_usage" else "memory"
            params["scale_factor"] = 2.0
        elif metric == "bandwidth_utilization":
            params["resource"] = "bandwidth"
            params["scale_factor"] = 1.5
        elif metric == "throughput_mbps":
            params["target_rate_mbps"] = max(50, int(anomaly.value * 0.5))
        elif metric in ("packet_loss", "error_rate"):
            params["service"] = "network_core"
        else:
            params["service"] = "default"

        return params
