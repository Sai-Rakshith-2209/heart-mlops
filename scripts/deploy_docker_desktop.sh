#!/usr/bin/env bash
# Deploy API + Prometheus + Grafana to Docker Desktop's built-in Kubernetes (Tasks 7 & 8).
# Prerequisite: Docker Desktop -> Settings -> Kubernetes -> Enable (cluster type: kubeadm).
# With kubeadm, the cluster shares Docker's image store, so a locally built image is
# usable directly (imagePullPolicy: IfNotPresent) and LoadBalancer services bind to localhost.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=heart-disease-api:1.0.0

ctx=$(kubectl config current-context)
if [ "$ctx" != "docker-desktop" ]; then
  echo "kubectl context is '$ctx', expected 'docker-desktop'." >&2
  echo "Run: kubectl config use-context docker-desktop" >&2
  exit 1
fi

docker build -t "$IMAGE" .

kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/
# Grafana admin password: random, stored only in a Secret (never in git).
kubectl -n monitoring get secret grafana-admin >/dev/null 2>&1 || \
  kubectl -n monitoring create secret generic grafana-admin \
    --from-literal=password="$(openssl rand -hex 12)"
kubectl apply -k k8s/monitoring

kubectl -n heart rollout status deploy/heart-api --timeout=180s
kubectl -n monitoring rollout status deploy/prometheus --timeout=180s
kubectl -n monitoring rollout status deploy/grafana --timeout=180s

echo; echo "=== Resources"
kubectl get deploy,pods,svc -n heart -o wide
kubectl get pods -n monitoring

echo; echo "=== Prediction through the LoadBalancer (localhost:80)"
for _ in $(seq 1 30); do curl -sf localhost/ready >/dev/null && break; sleep 2; done
curl -s localhost/ready; echo
curl -s -X POST localhost/predict -H 'Content-Type: application/json' -d @sample_request.json; echo

cat <<'MSG'

Next (each in its own terminal):
  kubectl -n monitoring port-forward svc/prometheus 9090:9090   # http://localhost:9090/targets
  kubectl -n monitoring port-forward svc/grafana 3000:3000      # http://localhost:3000  (user: admin)
  Grafana password:
    kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.password}' | base64 -d; echo
  Traffic for the dashboard:
    python scripts/load_test.py --url http://localhost --requests 300
MSG
