"""Vercel Serverless Function: /api/health"""

from http.server import BaseHTTPRequestHandler
import json
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from lib.inference import engine


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        response = {
            "status": "ok",
            "version": "1.0.0",
            "model_status": "trained" if engine.ready else "untrained",
            "platform": "vercel-serverless",
        }

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())

    def log_message(self, format, *args):
        pass
