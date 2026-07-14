"""Tests for self-healing module."""

import pytest

from src.anomaly_detection.detector import AnomalyResult, Severity
from src.self_healing.agent import SelfHealingAgent
from src.self_healing.strategies import (
    ALL_STRATEGIES,
    CircuitBreaker,
    RemediationStatus,
    ResourceScaling,
    TrafficReroute,
)


def _make_anomaly(metric: str = "latency_ms", severity: str = "high") -> AnomalyResult:
    return AnomalyResult(
        timestamp="2026-01-01T00:00:00Z",
        node_id="node-001",
        metric_name=metric,
        value=120.0,
        severity=Severity(severity),
        reconstruction_error=1.5,
        prediction_error=0.8,
        is_anomaly=True,
        details={"metric_name": metric, "severity": severity},
    )


class TestStrategies:
    def test_all_strategies_registered(self):
        assert len(ALL_STRATEGIES) >= 5

    def test_traffic_reroute_handles_high_latency(self):
        strategy = TrafficReroute()
        anomaly = _make_anomaly("latency_ms", "high")
        assert strategy.can_handle(anomaly.details) is True

    def test_resource_scaling_handles_cpu(self):
        strategy = ResourceScaling()
        anomaly = _make_anomaly("cpu_usage", "medium")
        assert strategy.can_handle(anomaly.details) is True

    def test_circuit_breaker_only_critical(self):
        strategy = CircuitBreaker()
        assert strategy.can_handle({"severity": "critical"}) is True
        assert strategy.can_handle({"severity": "low"}) is False


class TestSelfHealingAgent:
    @pytest.mark.asyncio
    async def test_heal_disabled_returns_log_only(self):
        agent = SelfHealingAgent(enabled=False)
        anomaly = _make_anomaly()
        event = await agent.heal(anomaly)
        assert event.success is False
        assert "disabled" in event.resolution.lower()

    @pytest.mark.asyncio
    async def test_heal_enabled_executes_strategy(self):
        agent = SelfHealingAgent(enabled=True)
        anomaly = _make_anomaly("latency_ms", "critical")
        event = await agent.heal(anomaly)
        assert event.action is not None
        assert event.action.status in (RemediationStatus.SUCCESS, RemediationStatus.FAILED)

    def test_get_stats(self):
        agent = SelfHealingAgent()
        stats = agent.get_stats()
        assert "total_events" in stats
        assert "strategies_available" in stats
