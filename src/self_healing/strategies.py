"""Remediation strategies for self-healing network operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class RemediationStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    SUCCESS = "success"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


@dataclass
class RemediationAction:
    strategy_name: str
    target_node: str
    parameters: dict
    status: RemediationStatus = RemediationStatus.PENDING
    result: str = ""
    rollback_data: dict | None = None


class RemediationStrategy(ABC):
    """Base class for all remediation strategies."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @abstractmethod
    def can_handle(self, anomaly_details: dict) -> bool: ...

    @abstractmethod
    async def execute(self, action: RemediationAction) -> RemediationAction: ...

    async def rollback(self, action: RemediationAction) -> RemediationAction:
        """Override to implement rollback logic."""
        action.status = RemediationStatus.ROLLED_BACK
        return action


class TrafficReroute(RemediationStrategy):
    """Reroute traffic away from degraded node to healthy neighbors."""

    @property
    def name(self) -> str:
        return "traffic_reroute"

    @property
    def description(self) -> str:
        return "Reroute traffic from degraded node to healthy neighbors"

    def can_handle(self, anomaly_details: dict) -> bool:
        severity = anomaly_details.get("severity", "")
        metric = anomaly_details.get("metric_name", "")
        return severity in ("high", "critical") and metric in (
            "packet_loss",
            "latency_ms",
            "error_rate",
        )

    async def execute(self, action: RemediationAction) -> RemediationAction:
        action.status = RemediationStatus.IN_PROGRESS
        node = action.target_node
        logger.info("Rerouting traffic away from node %s", node)

        action.result = f"Traffic rerouted from {node} to backup path"
        action.status = RemediationStatus.SUCCESS
        return action


class ResourceScaling(RemediationStrategy):
    """Scale resources up/down based on utilization anomalies."""

    @property
    def name(self) -> str:
        return "resource_scaling"

    @property
    def description(self) -> str:
        return "Scale CPU/memory/bandwidth resources based on demand"

    def can_handle(self, anomaly_details: dict) -> bool:
        metric = anomaly_details.get("metric_name", "")
        return metric in ("cpu_usage", "memory_usage", "bandwidth_utilization", "connection_count")

    async def execute(self, action: RemediationAction) -> RemediationAction:
        action.status = RemediationStatus.IN_PROGRESS
        resource = action.parameters.get("resource", "cpu")
        scale_factor = action.parameters.get("scale_factor", 1.5)
        logger.info("Scaling %s by %.1fx on node %s", resource, scale_factor, action.target_node)

        action.result = f"Scaled {resource} by {scale_factor}x on {action.target_node}"
        action.status = RemediationStatus.SUCCESS
        return action


class ServiceRestart(RemediationStrategy):
    """Gracefully restart services on affected nodes."""

    @property
    def name(self) -> str:
        return "service_restart"

    @property
    def description(self) -> str:
        return "Gracefully restart degraded services"

    def can_handle(self, anomaly_details: dict) -> bool:
        severity = anomaly_details.get("severity", "")
        return severity in ("medium", "high")

    async def execute(self, action: RemediationAction) -> RemediationAction:
        action.status = RemediationStatus.IN_PROGRESS
        service = action.parameters.get("service", "default")
        logger.info("Restarting service %s on node %s", service, action.target_node)

        action.result = f"Service {service} restarted on {action.target_node}"
        action.status = RemediationStatus.SUCCESS
        return action


class RateLimiter(RemediationStrategy):
    """Apply rate limiting to prevent cascade failures."""

    @property
    def name(self) -> str:
        return "rate_limiting"

    @property
    def description(self) -> str:
        return "Apply adaptive rate limiting to prevent cascade failures"

    def can_handle(self, anomaly_details: dict) -> bool:
        metric = anomaly_details.get("metric_name", "")
        return metric in ("throughput_mbps", "retransmit_rate", "queue_depth")

    async def execute(self, action: RemediationAction) -> RemediationAction:
        action.status = RemediationStatus.IN_PROGRESS
        rate = action.parameters.get("target_rate_mbps", 100)
        logger.info("Applying rate limit %d Mbps on node %s", rate, action.target_node)

        action.result = f"Rate limited to {rate} Mbps on {action.target_node}"
        action.status = RemediationStatus.SUCCESS
        return action


class CircuitBreaker(RemediationStrategy):
    """Isolate failing nodes to prevent cascading failures."""

    @property
    def name(self) -> str:
        return "circuit_breaker"

    @property
    def description(self) -> str:
        return "Open circuit breaker to isolate critically failing nodes"

    def can_handle(self, anomaly_details: dict) -> bool:
        severity = anomaly_details.get("severity", "")
        return severity == "critical"

    async def execute(self, action: RemediationAction) -> RemediationAction:
        action.status = RemediationStatus.IN_PROGRESS
        logger.warning("Opening circuit breaker on node %s", action.target_node)

        action.result = f"Circuit breaker opened on {action.target_node}"
        action.status = RemediationStatus.SUCCESS
        return action


ALL_STRATEGIES: list[RemediationStrategy] = [
    TrafficReroute(),
    ResourceScaling(),
    ServiceRestart(),
    RateLimiter(),
    CircuitBreaker(),
]
