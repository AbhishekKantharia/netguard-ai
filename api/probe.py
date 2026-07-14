"""Vercel Serverless Function: /api/probe

Probes real internet endpoints to measure live network telemetry,
then runs anomaly detection on the gathered metrics.
No manual input required — fully automatic.
"""

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import sys
import os
import time
import urllib.request
import urllib.error
import ssl
import statistics
import uuid
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.inference import engine, METRIC_NAMES

PROBE_TARGETS = [
    {"name": "Google", "url": "https://www.google.com/generate_204", "size": 0},
    {"name": "Cloudflare", "url": "https://1.1.1.1/cdn-cgi/trace", "size": 200},
    {"name": "GitHub", "url": "https://api.github.com", "size": 500},
    {"name": "AWS", "url": "https://aws.amazon.com", "size": 5000},
    {"name": "Microsoft", "url": "https://www.microsoft.com", "size": 5000},
    {"name": "Cloudflare DNS", "url": "https://dns.google/resolve?name=google.com&type=A", "size": 200},
]

STRATEGY_MAP = {
    "cpu_usage": ("resource_scaling", "Scaled cpu by 2.0x"),
    "memory_usage": ("resource_scaling", "Scaled memory by 1.5x"),
    "bandwidth_utilization": ("resource_scaling", "Scaled bandwidth by 2.0x"),
    "packet_loss": ("traffic_reroute", "Traffic rerouted to backup path"),
    "latency_ms": ("traffic_reroute", "Traffic rerouted to low-latency path"),
    "error_rate": ("service_restart", "Service restarted to clear errors"),
    "throughput_mbps": ("rate_limiting", "Rate limited to prevent congestion"),
    "retransmit_rate": ("rate_limiting", "Rate limited to reduce retransmits"),
    "queue_depth": ("rate_limiting", "Rate limited to drain queue"),
    "connection_count": ("circuit_breaker", "Circuit breaker opened to isolate"),
    "jitter_ms": ("service_restart", "Service restarted to stabilize timing"),
    "temperature_celsius": ("resource_scaling", "Scaled cooling resources"),
}


def probe_endpoint(target, timeout=5):
    """Probe a single endpoint and return latency, success, and throughput."""
    ctx = ssl.create_default_context()
    req = urllib.request.Request(target["url"], headers={"User-Agent": "NetGuard-AI/1.0"})
    start = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout, context=ctx)
        body = resp.read(1024)
        elapsed = (time.monotonic() - start) * 1000
        status = resp.getcode()
        return {
            "name": target["name"],
            "latency_ms": round(elapsed, 2),
            "success": True,
            "status": status,
            "bytes": len(body),
            "error": None,
        }
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        elapsed = (time.monotonic() - start) * 1000
        return {
            "name": target["name"],
            "latency_ms": round(elapsed, 2),
            "success": False,
            "status": 0,
            "bytes": 0,
            "error": str(e),
        }


def compute_metrics(results):
    """Convert probe results into the 12-metric network telemetry format."""
    latencies = [r["latency_ms"] for r in results]
    successes = [r for r in results if r["success"]]
    failures = [r for r in results if not r["success"]]
    total_bytes = sum(r["bytes"] for r in successes)

    avg_latency = statistics.mean(latencies) if latencies else 50.0
    jitter = statistics.stdev(latencies) if len(latencies) > 1 else 0.0
    loss_pct = (len(failures) / len(results)) * 100 if results else 0.0

    success_count = len(successes)
    total_count = len(results)
    error_rate = (len(failures) / total_count) * 100 if total_count else 0.0

    throughput = (total_bytes / (avg_latency / 1000)) / 1_000_000 if avg_latency > 0 else 0.0

    return {
        "cpu_usage": min(99.0, max(1.0, 30.0 + jitter * 2)),
        "memory_usage": min(99.0, max(10.0, 45.0 + len(results) * 2)),
        "bandwidth_utilization": min(99.0, max(1.0, min(throughput / 5, 95.0))),
        "packet_loss": round(loss_pct, 2),
        "latency_ms": round(avg_latency, 2),
        "jitter_ms": round(jitter, 2),
        "error_rate": round(error_rate, 2),
        "throughput_mbps": round(max(0.1, throughput), 2),
        "connection_count": total_count * 150 + success_count * 800,
        "retransmit_rate": round(min(loss_pct * 1.5, 5.0), 2),
        "queue_depth": round(max(0, jitter * 3), 1),
        "temperature_celsius": round(42.0 + avg_latency * 0.1 + jitter * 0.5, 1),
    }


def simulate_healing(metrics, severity, node_id):
    dominant = max(metrics, key=lambda k: abs(metrics[k])) if metrics else None
    if severity not in ("high", "critical"):
        return None
    strategy, resolution = STRATEGY_MAP.get(dominant, ("resource_scaling", "Applied default remediation"))
    return {
        "event_id": str(uuid.uuid4()),
        "strategy": strategy,
        "status": "success",
        "resolution": f"{resolution} on {node_id}",
        "success": True,
    }


class handler(BaseHTTPRequestHandler):

    def do_GET(self):
        try:
            node_id = "probe-" + str(hash(str(time.monotonic_ns())) % 9999).replace("-", "")[:4]

            probe_results = [probe_endpoint(t) for t in PROBE_TARGETS]
            metrics = compute_metrics(probe_results)

            result = engine.detect(metrics)
            result["node_id"] = node_id
            result["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            result["metric_name"] = max(metrics, key=lambda k: abs(metrics[k]))
            result["value"] = metrics.get(result["metric_name"], 0.0)

            if result.get("is_anomaly"):
                result["healing_event"] = simulate_healing(metrics, result["severity"], node_id)

            result["probe_results"] = [
                {"name": r["name"], "latency_ms": r["latency_ms"], "success": r["success"], "error": r["error"]}
                for r in probe_results
            ]

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        except Exception as e:
            tb = traceback.format_exc()
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass
