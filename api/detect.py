"""Vercel Serverless Function: /api/detect"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.inference import engine, METRIC_NAMES


class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        data = json.loads(body)

        node_id = data.get("node_id", "unknown")
        metrics = data.get("metrics", {})
        timestamp = data.get("timestamp", "")
        metrics["_node_id"] = node_id

        result = engine.detect(metrics)
        result["node_id"] = node_id
        result["timestamp"] = timestamp
        result["metric_name"] = max(metrics, key=lambda k: abs(metrics[k])) if metrics else "unknown"
        result["value"] = metrics.get(result["metric_name"], 0.0)

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(result).encode())

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        pass
