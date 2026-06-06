#!/bin/bash
# Railway entrypoint: use $PORT env or default to 8080
PORT=${PORT:-8080}
echo "Starting PepScope on port $PORT"
exec uvicorn app:app --host 0.0.0.0 --port $PORT
