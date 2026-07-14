# Changelog

All notable changes to NetGuard AI will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-14

### 🎉 Initial Release

#### Added

##### Core Features
- **Anomaly Detection Engine**
  - PyTorch Autoencoder for reconstruction-based anomaly detection
  - PyTorch LSTM Predictor for temporal anomaly detection
  - Youden's J statistic for optimal threshold selection
  - Model status reporting (untrained/training/trained/degraded)
  - Real-time inference with <10ms latency

- **Self-Healing Agent**
  - 5 automated healing strategies:
    - Traffic Reroute (high latency + packet loss)
    - Resource Scaling (high CPU/memory usage)
    - Service Restart (critical severity + service down)
    - Rate Limiting (moderate anomaly + traffic spike)
    - Circuit Breaker (severe anomaly cascade)
  - Orchestrator with strategy selection logic
  - Event logging and history tracking

- **API Server**
  - FastAPI backend with async support
  - REST endpoints for detection, training, and monitoring
  - WebSocket support for real-time metric ingestion
  - Background model training (non-blocking)
  - Interactive Swagger API documentation

- **Frontend Dashboard**
  - React-based real-time monitoring UI
  - Network health visualization
  - Healing event history display
  - Live metric streaming via WebSocket

##### Deployment
- **Railway**: Production backend deployment with PyTorch
- **Vercel**: Frontend + ONNX serverless functions
- **Docker**: Containerized deployment support
- **Fly.io**: Ready for deployment (pending billing)

##### Models
- Exported ONNX models for lightweight inference
- Autoencoder ONNX model
- LSTM Predictor ONNX model

##### Testing
- 19 comprehensive tests (19/19 passing)
  - Anomaly detection tests (9)
  - Self-healing tests (6)
  - API endpoint tests (3)

##### Documentation
- Comprehensive README with architecture diagrams
- API reference documentation
- Contributing guidelines
- Security policy
- Changelog

#### Technical Details
- Python 3.10+ compatibility
- PyTorch 2.x for model training
- Pydantic v2 for data validation
- ruff for linting and formatting
- 100-character line length
- Type hints throughout codebase

#### Performance
- Detection accuracy: 61.5% (degraded status — retraining in progress)
- F1 Score: 0.5969
- Score separation: 2,561,247
- Inference latency: <10ms per prediction
- Model auto-threshold: 434,294

---

## [0.1.0] - 2026-07-14

### Added
- Initial project structure
- Core anomaly detection module
- Self-healing agent
- FastAPI server
- Basic test suite
- Docker configuration
- Vercel deployment setup

---

[1.0.0]: https://github.com/AbhishekKantharia/netguard-ai/releases/tag/v1.0.0
[0.1.0]: https://github.com/AbhishekKantharia/netguard-ai/releases/tag/v0.1.0
