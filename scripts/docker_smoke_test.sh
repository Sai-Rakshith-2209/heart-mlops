#!/usr/bin/env bash
# Build the serving image, run it, and prove it serves predictions (Task 6 evidence).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=heart-disease-api:1.0.0

docker build -t "$IMAGE" .
docker rm -f heart-api-smoke >/dev/null 2>&1 || true
docker run -d --name heart-api-smoke -p 127.0.0.1:8000:8000 "$IMAGE"
trap 'docker rm -f heart-api-smoke >/dev/null' EXIT

for _ in $(seq 1 30); do curl -sf localhost:8000/ready >/dev/null && break; sleep 1; done
echo "--- /ready";   curl -s localhost:8000/ready; echo
echo "--- /predict"; curl -s -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' -d @sample_request.json; echo
echo "--- invalid input (expect 422)"
curl -s -o /dev/null -w '%{http_code}\n' -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' -d '{"age": 999}'
echo "--- container health"; docker inspect --format '{{.State.Health.Status}}' heart-api-smoke
echo "--- container logs";   docker logs heart-api-smoke | tail -5
