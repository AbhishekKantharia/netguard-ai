<div align="center">

# 🛡️ NetGuard AI

### Predictive Network Self-Healing System for AI-Native Telecom

[![GitHub Stars](https://img.shields.io/github/stars/AbhishekKantharia/netguard-ai?style=flat-square)](https://github.com/AbhishekKantharia/netguard-ai/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/AbhishekKantharia/netguard-ai?style=flat-square)](https://github.com/AbhishekKantharia/netguard-ai/network/members)
[![GitHub Issues](https://img.shields.io/github/issues/AbhishekKantharia/netguard-ai?style=flat-square)](https://github.com/AbhishekKantharia/netguard-ai/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/AbhishekKantharia/netguard-ai?style=flat-square)](https://github.com/AbhishekKantharia/netguard-ai/pulls)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg?style=flat-square)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg?style=flat-square)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688.svg?style=flat-square)](https://fastapi.tiangolo.com/)
[![Tests](https://img.shields.io/badge/tests-19%20passing-brightgreen.svg?style=flat-square)]()
[![Ruff](https://img.shields.io/badge/lint-ruff-clean-brightgreen.svg?style=flat-square)]()

---

**Built for the Deutsche Telekom Digital Labs (DTDL) 2026 T Challenge**

NetGuard AI uses machine learning to detect network anomalies before they cause outages and automatically triggers self-healing remediations — reducing MTTR from hours to seconds.

[🚀 Live Frontend](https://netguard-ai-pearl.vercel.app) · [📡 Live API](https://netguard-ai-production.up.railway.app) · [📖 API Docs](https://netguard-ai-production.up.railway.app/docs) · [💻 Source Code](https://github.com/AbhishekKantharia/netguard-ai)

---

</div>

## 🎯 Problem Statement

Network outages cost the telecom industry **billions annually** and affect **millions of users**. Traditional reactive maintenance is:

- **Slow** — MTTR (Mean Time To Repair) measured in hours
- **Manual** — Requires human intervention for every incident
- **Reactive** — Only responds after failures occur
- **Incomplete** — Fails to prevent cascading failures

DT's 2026 T Challenge theme of **"AI-Native Telco"** demands solutions that embed intelligence directly into network operations.

## 💡 Solution

NetGuard AI combines predictive anomaly detection with automated self-healing:

```
┌─────────────────────────────────────────────────────────────────┐
│                        NetGuard AI                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐     ┌──────────────────────────────────┐    │
│   │  Network     │────▶│    Anomaly Detection Engine       │    │
│   │  Telemetry   │     │    ┌──────────┐  ┌──────────┐    │    │
│   │  (Metrics)   │     │    │Autoencdr │  │LSTM Pred │    │    │
│   └──────────────┘     │    └──────────┘  └──────────┘    │    │
│                         └───────────────┬──────────────────┘    │
│                                         │                       │
│                         ┌───────────────▼──────────────────┐    │
│                         │    Self-Healing Agent             │    │
│                         │    • Traffic Reroute              │    │
│                         │    • Resource Scaling             │    │
│                         │    • Service Restart              │    │
│                         │    • Rate Limiting                │    │
│                         │    • Circuit Breaker              │    │
│                         └──────────────────────────────────┘    │
│                                                                 │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │  FastAPI Server  ·  WebSocket  ·  React Dashboard UI     │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🔧 Core Components

| Component | Purpose | Technology |
|-----------|---------|------------|
| **Autoencoder** | Learns normal network behavior; flags deviations via reconstruction error | PyTorch |
| **LSTM Predictor** | Forecasts next-step metrics; large prediction errors signal emerging issues | PyTorch |
| **Self-Healing Agent** | Orchestrates automatic remediation strategies | Python |
| **FastAPI Server** | REST API + WebSocket for real-time metric ingestion | FastAPI |
| **React Dashboard** | Live visualization of network health and healing events | React |
| **ONNX Runtime** | Lightweight inference for edge/serverless deployment | ONNX |

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- pip or poetry

### Installation

```bash
# Clone the repository
git clone https://github.com/AbhishekKantharia/netguard-ai.git
cd netguard-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"

# Run the server
python main.py
```

Server runs at `http://localhost:8000`. Interactive API docs at `/docs`.

### Docker

```bash
# Build and run with Docker
docker compose up --build

# Or use Docker directly
docker build -t netguard-ai .
docker run -p 8000:8000 netguard-ai
```

## 📡 API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check with model status |
| `POST` | `/api/v1/detect` | Run anomaly detection on metrics |
| `POST` | `/api/v1/batch-detect` | Batch detection for multiple nodes |
| `POST` | `/api/v1/train` | Train/retrain models on synthetic data |
| `GET` | `/api/v1/model/status` | Get model training status and metrics |
| `GET` | `/api/v1/nodes` | List all monitored network nodes |
| `GET` | `/api/v1/stats` | System-wide statistics |
| `GET` | `/api/v1/events` | Recent healing events |
| `WS` | `/ws/metrics` | Real-time metric ingestion via WebSocket |

### Example: Detect + Heal

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

### Example: Batch Detection

```python
import requests

response = requests.post("http://localhost:8000/api/v1/batch-detect", json={
    "metrics_batch": [
        {"node_id": "node-001", "metrics": {"cpu_usage": 95.0, "memory_usage": 88.0}},
        {"node_id": "node-002", "metrics": {"cpu_usage": 45.0, "memory_usage": 62.0}},
    ]
})

results = response.json()
for r in results["results"]:
    print(f"{r['node_id']}: anomaly={r['is_anomaly']}, severity={r['severity']}")
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --tb=short

# Lint check
ruff check src/ tests/

# Format check
ruff format --check src/ tests/
```

**Test Results: 19/19 passing** ✅

- `test_anomaly_detection.py` — 9 tests for detector, model training, evaluation
- `test_self_healing.py` — 6 tests for healing strategies and agent orchestration
- `test_api.py` — 3 tests for API endpoints and health checks

## 🏗️ Project Structure

```
netguard-ai/
├── src/
│   ├── anomaly_detection/
│   │   ├── detector.py      # Core anomaly detection engine
│   │   ├── models.py        # PyTorch Autoencoder + LSTM models
│   │   └── data_generator.py # Synthetic telemetry data generator
│   ├── self_healing/
│   │   ├── agent.py         # Self-healing orchestrator
│   │   └── strategies.py    # 5 healing strategies
│   └── api/
│       ├── app.py           # FastAPI application
│       └── schemas.py       # Pydantic request/response models
├── tests/
│   ├── test_anomaly_detection.py
│   ├── test_self_healing.py
│   └── test_api.py
├── api/                     # Vercel serverless functions
│   ├── detect.py
│   ├── health.py
│   └── stats.py
├── lib/                     # Shared libraries
│   └── inference.py         # ONNX inference engine
├── public/                  # Frontend
│   ├── index.html           # React dashboard
│   └── models/              # Exported ONNX models
├── scripts/
│   └── export_onnx.py       # PyTorch → ONNX export
├── docs/                    # Documentation
├── main.py                  # Entry point
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose config
├── requirements.txt         # Python dependencies
├── pyproject.toml           # Project configuration
└── vercel.json              # Vercel deployment config
```

## 🧠 Model Details

### Autoencoder Architecture

```
Input (5 features) → Encoder → Bottleneck (8 dims) → Decoder → Output (5 features)
```

- Learns to reconstruct normal network behavior
- High reconstruction error → anomaly detected
- Threshold: Youden's J statistic (optimal sensitivity + specificity)

### LSTM Predictor Architecture

```
Input (5 features × 20 timesteps) → LSTM (32 hidden) → Linear → Output (5 features)
```

- Forecasts next-step network metrics
- Large prediction error → emerging issue detected
- Captures temporal dependencies in metric sequences

### Self-Healing Strategies

| Strategy | Trigger Condition | Action |
|----------|-------------------|--------|
| **Traffic Reroute** | High latency + packet loss | Redirect traffic to backup path |
| **Resource Scaling** | High CPU/memory usage | Scale up resources |
| **Service Restart** | Critical severity + service down | Restart affected service |
| **Rate Limiting** | Moderate anomaly + traffic spike | Apply rate limits |
| **Circuit Breaker** | Severe anomaly cascade | Isolate affected segment |

## 🌐 Deployments

| Service | URL | Description |
|---------|-----|-------------|
| **Frontend** | [netguard-ai-pearl.vercel.app](https://netguard-ai-pearl.vercel.app) | React dashboard with live metrics |
| **Backend** | [netguard-ai-production.up.railway.app](https://netguard-ai-production.up.railway.app) | FastAPI server with PyTorch |
| **API Docs** | [netguard-ai-production.up.railway.app/docs](https://netguard-ai-production.up.railway.app/docs) | Interactive Swagger UI |
| **GitHub** | [AbhishekKantharia/netguard-ai](https://github.com/AbhishekKantharia/netguard-ai) | Source code repository |

## 🔗 Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3.10+, FastAPI, Uvicorn |
| **ML Framework** | PyTorch 2.x |
| **Inference** | ONNX Runtime (Vercel) |
| **Validation** | Pydantic v2 |
| **Testing** | pytest, pytest-asyncio |
| **Linting** | ruff |
| **Frontend** | React, HTML5, CSS3, WebSocket |
| **Deployment** | Railway (backend), Vercel (frontend), Docker |
| **CI/CD** | GitHub Actions (planned) |

## 🎯 DTDL Challenge Alignment

| Challenge Theme | How NetGuard AI Addresses It |
|-----------------|------------------------------|
| **AI-Native Telco** | ML models embedded directly in network operations loop |
| **RAN Guardian Parallels** | Predictive detection + automated response |
| **Scalability** | WebSocket support for millions of concurrent metric streams |
| **Real Business Impact** | Reduces MTTR from hours to seconds |
| **EU AI Act Compliance** | Transparent decision-making with explainable anomaly scores |
| **Open Source** | MIT licensed, community-driven development |

## 📊 Performance

- **Detection Accuracy**: 61.5% (degraded status — retraining in progress)
- **F1 Score**: 0.5969
- **Score Separation**: 2,561,247 (anomaly vs normal)
- **Inference Latency**: <10ms per prediction
- **Model Auto-threshold**: 434,294 (Youden's J optimal)
- **Test Coverage**: 19/19 tests passing

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'feat: add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

## 🙏 Acknowledgments

- **Deutsche Telekom Digital Labs** for the challenge opportunity
- **PyTorch** for the ML framework
- **FastAPI** for the backend framework
- **Railway** and **Vercel** for deployment infrastructure

---

<div align="center">

**Built with ❤️ for AI-Native Telecom**

[⬆ Back to Top](#-netguard-ai)

</div>
