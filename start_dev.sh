#!/usr/bin/env bash
set -e

export PATH="$HOME/.local/bin:$PATH"

echo "=========================================================="
echo "📰 360° News Feedback System — SIH1329 Situation Room"
echo "=========================================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

# 1. Check Python virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
    .venv/bin/pip install fastapi uvicorn pydantic python-multipart
fi

# 2. Check frontend dependencies
if [ ! -d "web/node_modules" ]; then
    echo "Installing frontend dependencies..."
    cd web && npm install && cd ..
fi

echo "Starting FastAPI Backend on http://localhost:8000..."
.venv/bin/uvicorn api.main:app --port 8000 &
BACKEND_PID=$!

echo "Starting Vite + React + TypeScript Frontend on http://localhost:5173..."
cd web && npm run dev &
FRONTEND_PID=$!

cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    exit 0
}

trap cleanup INT TERM

wait
