# NetGuard AI

**Predictive Network Self-Healing System for AI-Native Telecom**

Built for the Deutsche Telekom Digital Labs (DTDL) challenge. NetGuard AI uses machine learning to detect network anomalies before they cause outages and automatically triggers self-healing remediations.

## Problem Statement

Network outages cost the telecom industry billions annually and affect millions of users. Traditional reactive maintenance is slow, manual, and fails to prevent cascading failures. DT's 2026 T Challenge theme of "AI-Native Telco" demands solutions that embed intelligence directly into network operations.

## Solution

NetGuard AI combines:

| Component | Purpose |
|-----------|---------|
| **Autoencoder** | Learns normal network behavior; flags deviations via reconstruction error |
| **LSTM Predictor** | Forecasts next-step metrics; large prediction errors signal emerging issues |
| **Self-Healing Agent** | Orchestrates automatic remediation (reroute, scale, restart, circuit-breaker) |
| **Real-time API** | FastAPI endpoints + WebSocket for live metric ingestion |

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    NetGuard AI                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   ┌──────────────┐     ┌────────────────────────────┐  │
│   │  Network     │────▶│  Anomaly Detection Engine   │  │
│   │  Telemetry   │     │  ┌──────────┐ ┌──────────┐ │  │
│   │  (Metrics)   │     │  │Autoencdr │ │LSTM Pred │ │  │
│   └──────────────┘     │  └──────────┘ └──────────┘ │  │
│                         └─────────────┬──────────────┘  │
│                                       │                 │
│                         ┌─────────────▼──────────────┐  │
│                         │  Self-Healing Agent         │  │
│                         │  • Traffic Reroute          │  │
│                         │  • Resource Scaling         │  │
│                         │  • Service Restart          │  │
│                         │  • Rate Limiting            │  │
│                         │  • Circuit Breaker          │  │
│                         └────────────────────────────┘  │
│                                                         │
│   ┌──────────────────────────────────────────────────┐  │
│   │  FastAPI Server  +  WebSocket  +  Dashboard UI   │  │
│   └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

```bash
# 1. Clone
git clone https://github.com/your-username/netguard-ai.git
cd netguard-ai

# 2. Create venv
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # macOS/Linux

# 3. Install deps
pip install -e ".[dev]"

# 4. Run server
python main.py
```

Server runs at `http://localhost:8000`. Docs at `/docs`.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/api/v1/detect` | Run anomaly detection on metrics |
| `POST` | `/api/v1/batch-detect` | Batch detection |
| `POST` | `/api/v1/train` | Train models on synthetic data |
| `GET` | `/api/v1/nodes` | List monitored nodes |
| `GET` | `/api/v1/stats` | System-wide statistics |
| `GET` | `/api/v1/events` | Recent healing events |
| `WS` | `/ws/metrics` | Real-time metric ingestion |

## Example: Detect + Heal

```python
import requests

response = requests.post("http://localhost:8000/api/v1/detect", json={
    "node_id": "node-001",
    "metrics": {
        "cpu_usage": 95.0,
        "memory_usage": 88.0,
        "latency_ms": 250.0,
        "packet_loss": 12.0,
    }
})

result = response.json()
print(f"Anomaly: {result['is_anomaly']}")
print(f"Severity: {result['severity']}")
if result["healing_event"]:
    print(f"Action: {result['healing_event']['strategy']}")
```

## Running Tests

```bash
pytest tests/ -v
```

## Tech Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn
- **ML**: PyTorch (Autoencoder + LSTM)
- **Validation**: Pydantic v2
- **Testing**: pytest, pytest-asyncio
- **Linting**: ruff

## How It Aligns with DTDL

- **AI-Native Telco**: Models embedded directly in network operations loop
- **RAN Guardian Parallels**: Predictive detection + automated response
- **Scalability**: WebSocket support for millions of concurrent metric streams
- **Real Business Impact**: Reduces MTTR from hours to seconds
- **EU AI Act Compliance**: Transparent decision-making with explainable anomaly scores

## License

MIT
