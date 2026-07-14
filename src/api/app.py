"""NetGuard AI - FastAPI application for network anomaly detection and self-healing."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from src.aggregator import NetworkAggregator
from src.api.schemas import (
    AnomalyResponse,
    HealthResponse,
    MetricsPayload,
    NodeStatus,
    SystemStats,
    TestMetricsResponse,
    TrainRequest,
    TrainResponse,
)

logger = logging.getLogger(__name__)

_start_time = time.time()
_aggregator = NetworkAggregator()


@asynccontextmanager
async def lifespan(app: FastAPI):
    import threading

    def _train():
        logger.info("Background: training and evaluating detector...")
        _aggregator.train(epochs=50, num_samples=2000)
        logger.info("Background: model status = %s", _aggregator.detector.status.value)

    threading.Thread(target=_train, daemon=True).start()
    logger.info("NetGuard AI starting up (training in background)")
    yield
    logger.info("NetGuard AI shutting down")


app = FastAPI(
    title="NetGuard AI",
    description="Predictive Network Self-Healing System for AI-Native Telecom",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    detector = _aggregator.detector
    test_metrics = None
    if detector.status.value in ("trained", "degraded"):
        test_metrics = TestMetricsResponse(**detector.test_metrics.to_dict())
    return HealthResponse(
        status="ok",
        version="0.2.0",
        model_status=detector.status.value,
        test_metrics=test_metrics,
    )


@app.post("/api/v1/detect", response_model=AnomalyResponse)
async def detect_anomaly(payload: MetricsPayload):
    """Run anomaly detection on incoming network metrics."""
    result = _aggregator.process_metrics(
        node_id=payload.node_id,
        metrics=payload.metrics,
        timestamp=payload.timestamp,
    )
    return result


@app.post("/api/v1/batch-detect")
async def batch_detect(payloads: list[MetricsPayload]):
    """Run anomaly detection on a batch of metric observations."""
    results = []
    for p in payloads:
        results.append(
            _aggregator.process_metrics(
                node_id=p.node_id,
                metrics=p.metrics,
                timestamp=p.timestamp,
            )
        )
    return {"results": results, "count": len(results)}


@app.post("/api/v1/train", response_model=TrainResponse)
async def train_detector(request: TrainRequest):
    """Train the anomaly detection models and evaluate on held-out test data."""
    import asyncio

    loop = asyncio.get_event_loop()
    metrics = await loop.run_in_executor(
        None,
        lambda: _aggregator.train(
            epochs=request.epochs,
            num_samples=request.num_samples,
            test_ratio=request.test_ratio,
        ),
    )
    return TrainResponse(
        status=metrics["status"],
        samples=metrics["samples"],
        epochs=metrics["epochs"],
        ae_initial_loss=metrics["ae_initial_loss"],
        ae_final_loss=metrics["ae_final_loss"],
        lstm_final_loss=metrics["lstm_final_loss"],
        test_metrics=TestMetricsResponse(**{
            k: v for k, v in metrics.items()
            if k not in ("status", "samples", "epochs", "ae_initial_loss",
                         "ae_final_loss", "lstm_final_loss")
        }),
        message=f"Training complete. Model status: {metrics['status']}",
    )


@app.get("/api/v1/nodes", response_model=list[NodeStatus])
async def list_nodes():
    """List all monitored network nodes and their status."""
    return _aggregator.get_node_statuses()


@app.get("/api/v1/stats", response_model=SystemStats)
async def get_stats():
    """Get system-wide statistics."""
    return _aggregator.get_system_stats()


@app.get("/api/v1/events")
async def get_events(limit: int = 50):
    """Get recent healing events."""
    events = _aggregator.get_healing_events(limit)
    return {"events": events, "count": len(events)}


@app.websocket("/ws/metrics")
async def websocket_metrics(ws: WebSocket):
    """WebSocket endpoint for real-time metric ingestion."""
    await ws.accept()
    logger.info("WebSocket client connected")
    try:
        while True:
            data = await ws.receive_json()
            payload = MetricsPayload(**data)
            result = _aggregator.process_metrics(
                node_id=payload.node_id,
                metrics=payload.metrics,
                timestamp=payload.timestamp,
            )
            await ws.send_json(result)
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
