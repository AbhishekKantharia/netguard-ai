# Contributing to NetGuard AI

Thank you for your interest in contributing to NetGuard AI! This document provides guidelines and instructions for contributing.

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Issue Guidelines](#issue-guidelines)

## Code of Conduct

We are committed to providing a welcoming and inclusive experience for everyone. Please be respectful and constructive in all interactions.

## Getting Started

1. **Fork** the repository
2. **Clone** your fork locally
3. **Set up** the development environment (see below)
4. **Create** a feature branch
5. **Make** your changes
6. **Test** your changes
7. **Submit** a pull request

## Development Setup

### Prerequisites

- Python 3.10 or higher
- pip
- Git

### Installation

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/netguard-ai.git
cd netguard-ai

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install dependencies
pip install -e ".[dev]"

# Verify installation
pytest tests/ -v
```

### Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` with your preferred settings. Default values work for local development.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/AbhishekKantharia/netguard-ai/issues)
2. If not, create a new issue with:
   - Clear, descriptive title
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment details (OS, Python version, etc.)

### Suggesting Features

1. Check existing issues for similar suggestions
2. Create a new issue with the `feature-request` label
3. Describe the feature, its use case, and potential implementation

### Submitting Changes

1. **Branch naming**: Use descriptive names like `fix/detector-threshold` or `feature/new-strategy`
2. **Commit messages**: Follow [Conventional Commits](https://www.conventionalcommits.org/):
   - `feat:` for new features
   - `fix:` for bug fixes
   - `docs:` for documentation changes
   - `test:` for adding tests
   - `refactor:` for code refactoring
3. **Keep changes focused**: One logical change per pull request

## Code Style

### Python

- **Formatter/Linter**: We use [ruff](https://docs.astral.sh/ruff/)
- **Line length**: 100 characters max
- **Target version**: Python 3.10+

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Style Guidelines

- Use type hints for all function parameters and return values
- Use descriptive variable and function names
- Keep functions focused and concise
- Add docstrings for public functions and classes
- Use constants for magic numbers

### Example

```python
from typing import Optional

def detect_anomaly(
    metrics: dict[str, float],
    threshold: float = 0.85,
    node_id: Optional[str] = None,
) -> dict:
    """Detect anomalies in network metrics.

    Args:
        metrics: Network metric values (cpu_usage, memory_usage, etc.)
        threshold: Anomaly threshold (0-1)
        node_id: Optional node identifier

    Returns:
        Detection result with is_anomaly, severity, and score
    """
    # Implementation here
    pass
```

## Testing

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_anomaly_detection.py -v

# Run with coverage
pytest tests/ -v --tb=short

# Run specific test
pytest tests/test_anomaly_detection.py::test_detect_anomaly -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_<module>.py`
- Name test functions `test_<description>`
- Use descriptive test names
- Each test should test one thing
- Use fixtures for common setup

### Example Test

```python
import pytest
from src.anomaly_detection.detector import AnomalyDetector

class TestAnomalyDetector:
    """Tests for the AnomalyDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a fresh detector for each test."""
        return AnomalyDetector()

    def test_initial_status_is_untrained(self, detector):
        """Detector should start in untrained state."""
        assert detector.model_status.value == "untrained"

    def test_detect_returns_valid_result(self, detector):
        """Detection should return valid result structure."""
        metrics = {"cpu_usage": 50.0, "memory_usage": 60.0}
        result = detector.detect(metrics)
        assert "is_anomaly" in result
        assert "severity" in result
        assert "score" in result
```

## Pull Request Process

### Before Submitting

1. **Update documentation** if you changed APIs or added features
2. **Add tests** for new functionality
3. **Ensure all tests pass**: `pytest tests/ -v`
4. **Run linter**: `ruff check src/ tests/`
5. **Update CHANGELOG.md** with a summary of changes

### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added (if applicable)

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No new warnings generated
```

### Review Process

1. All PRs require at least one review
2. Address review feedback promptly
3. Squash commits before merging if requested
4. Ensure CI passes before merging

## Issue Guidelines

### Bug Reports

Use this template:

```markdown
**Describe the bug**
A clear description of the bug

**To Reproduce**
Steps to reproduce the behavior

**Expected behavior**
What you expected to happen

**Screenshots**
If applicable, add screenshots

**Environment**
- OS: [e.g., Windows 11]
- Python version: [e.g., 3.11.0]
- NetGuard AI version: [e.g., 1.0.0]
```

### Feature Requests

Use this template:

```markdown
**Is your feature request related to a problem?**
A clear description of the problem

**Describe the solution you'd like**
Your proposed solution

**Describe alternatives you've considered**
Other solutions you've thought about

**Additional context**
Any other context or screenshots
```

## Questions?

If you have questions about contributing, feel free to:

1. Open an issue with the `question` label
2. Start a discussion in the Discussions tab

Thank you for contributing to NetGuard AI! 🛡️
