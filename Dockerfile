FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] pydantic pandas scikit-learn \
    httpx python-dotenv websockets apscheduler

COPY . .

EXPOSE 8000

CMD ["python", "main.py"]
