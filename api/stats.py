"""Vercel Serverless Function: /api/stats"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.inference import engine

_start = time.time()


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        response = {
            "total_nodes": 0,
            "active_anomalies": 0,
            "total_healing_events": 0,
            "healing_success_rate": 0.0,
            "detector_fitted": engine.ready,
            "uptime_seconds": round(time.time() - _start, 1),
            "platform": "vercel-serverless",
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass
