"""Pydantic schemas for API request/response models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class MetricsPayload(BaseModel):
    node_id: str = Field(..., examples=["node-001"])
    timestamp: str = Field(default="", examples=["2026-07-14T20:00:00Z"])
    metrics: dict[str, float] = Field(
        ...,
        examples=[
            {
                "cpu_usage": 45.2,
                "memory_usage": 62.1,
                "latency_ms": 12.5,
                "packet_loss": 0.02,
            }
        ],
    )


class AnomalyResponse(BaseModel):
    node_id: str
    metric_name: str
    value: float
    severity: str
    reconstruction_error: float
    prediction_error: float
    is_anomaly: bool
    combined_score: float
    model_status: str = "untrained"
    healing_event: HealingEventResponse | None = None


class HealingEventResponse(BaseModel):
    event_id: str
    strategy: str
    status: str
    resolution: str
    success: bool


class TrainRequest(BaseModel):
    epochs: int = Field(default=50, ge=1, le=500)
    num_samples: int = Field(default=10000, ge=100, le=100000)
    test_ratio: float = Field(default=0.2, ge=0.05, le=0.5)


class TestMetricsResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_negatives: int
    false_positives: int
    false_negatives: int
    true_positives: int
    test_samples: int
    normal_samples: int
    anomaly_samples: int
    optimal_threshold: float
    mean_score_normal: float
    mean_score_anomaly: float
    score_separation: float


class TrainResponse(BaseModel):
    status: str
    samples: int
    epochs: int
    ae_initial_loss: float
    ae_final_loss: float
    lstm_final_loss: float
    test_metrics: TestMetricsResponse
    message: str


class NodeStatus(BaseModel):
    node_id: str
    status: str
    last_check: str
    anomaly_count_24h: int
    healing_events_24h: int


class SystemStats(BaseModel):
    total_nodes: int
    active_anomalies: int
    total_healing_events: int
    healing_success_rate: float
    model_status: str
    uptime_seconds: float


class HealthResponse(BaseModel):
    status: str
    version: str
    model_status: str
    test_metrics: TestMetricsResponse | None = None
