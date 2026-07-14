"""NetGuard AI entry point."""

import logging
import os
import sys

import uvicorn

from src.api.app import app

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    stream=sys.stdout,
)


def main():
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "src.api.app:app",
        host="0.0.0.0",
        port=port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
