"""Vercel Serverless Function: /api/detect

Supports both GET (query params) and POST (JSON body).
GET is the primary path since Vercel's BaseHTTPRequestHandler
doesn't reliably pipe POST bodies through rfile.
"""

from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import json
import sys
import os
import uuid
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.inference import engine, METRIC_NAMES


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

    def _handle_detect(self, data):
        try:
            node_id = data.get("node_id", "unknown")
            metrics = data.get("metrics", {})
            timestamp = data.get("timestamp", "")

            result = engine.detect(metrics)
            result["node_id"] = node_id
            result["timestamp"] = timestamp
            result["metric_name"] = max(metrics, key=lambda k: abs(metrics[k])) if metrics else "unknown"
            result["value"] = metrics.get(result["metric_name"], 0.0)

            if result.get("is_anomaly"):
                result["healing_event"] = simulate_healing(metrics, result["severity"], node_id)

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
            self.wfile.write(json.dumps({"error": str(e), "traceback": tb[-800:]}).encode("utf-8"))

    def do_GET(self):
        parsed = urlparse(self.path)
        params = parse_qs(parsed.query)

        data = {}
        for key in params:
            if key == "metrics":
                try:
                    data["metrics"] = json.loads(params[key][0])
                except (json.JSONDecodeError, IndexError):
                    pass
            else:
                data[key] = params[key][0]

        if not data.get("metrics"):
            data["metrics"] = {}
            for m in METRIC_NAMES:
                if m in params:
                    try:
                        data["metrics"][m] = float(params[m][0])
                    except (ValueError, IndexError):
                        pass

        self._handle_detect(data)

    def do_POST(self):
        try:
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            parsed = urlparse(self.path)
            params = parse_qs(parsed.query)
            data = {}
            for key in params:
                if key == "metrics":
                    try:
                        data["metrics"] = json.loads(params[key][0])
                    except (json.JSONDecodeError, IndexError):
                        pass
                else:
                    data[key] = params[key][0]

        self._handle_detect(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass
