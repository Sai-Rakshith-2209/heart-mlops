#!/usr/bin/env bash
# Build the serving image, run it, and prove it serves predictions (Task 6 evidence).
# Usage: ./scripts/docker_smoke_test.sh            # build + test
#        ./scripts/docker_smoke_test.sh --no-build # test an image that is already built (CI)
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=heart-disease-api:1.0.0
NAME=heart-api-smoke

if [ "${1:-}" != "--no-build" ]; then
  docker build -t "$IMAGE" .
fi

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" -p 127.0.0.1:8000:8000 "$IMAGE"
trap 'docker rm -f "$NAME" >/dev/null' EXIT

for _ in $(seq 1 30); do curl -sf localhost:8000/ready >/dev/null && break; sleep 1; done

echo "--- /ready"
curl -sf localhost:8000/ready; echo
echo "--- /predict"
curl -sf -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' -d @sample_request.json | tee /tmp/prediction.json; echo
python3 -c "
import json
r = json.load(open('/tmp/prediction.json'))
assert r['prediction'] in (0, 1) and 0.5 <= r['confidence'] <= 1, r
print('prediction contract OK')"

echo "--- invalid input (expect 422)"
code=$(curl -s -o /dev/null -w '%{http_code}' -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' -d '{"age": 999}')
echo "$code"; [ "$code" = "422" ]

echo "--- container health (waits for the HEALTHCHECK to pass)"
status=starting
for _ in $(seq 1 30); do
  status=$(docker inspect --format '{{.State.Health.Status}}' "$NAME")
  [ "$status" = "healthy" ] && break
  sleep 2
done
echo "$status"; [ "$status" = "healthy" ]

echo "--- runs as";        docker exec "$NAME" id
echo "--- image size";     docker image ls "$IMAGE" --format '{{.Size}}'
echo "--- container logs"; docker logs "$NAME" 2>&1 | tail -5
echo "SMOKE TEST PASSED"