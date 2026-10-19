#!/usr/bin/env bash
# Deploy API + Prometheus + Grafana to a local minikube cluster (Tasks 7 & 8).
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=heart-disease-api:1.0.0

minikube status >/dev/null 2>&1 || minikube start --cpus=4 --memory=6g
minikube addons enable ingress
minikube addons enable metrics-server

docker build -t "$IMAGE" .
minikube image load "$IMAGE"

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
kubectl get all,ingress,hpa,pdb -n heart
kubectl get pods -n monitoring

cat <<MSG

Next steps (each in its own terminal):
  minikube tunnel                                         # gives the LoadBalancer an IP
  kubectl -n heart get svc heart-api                      # EXTERNAL-IP column
  curl -X POST http://<EXTERNAL-IP>/predict -H 'Content-Type: application/json' -d @sample_request.json

  Ingress:  echo "\$(minikube ip) heart.local" | sudo tee -a /etc/hosts
            curl -X POST http://heart.local/predict -H 'Content-Type: application/json' -d @sample_request.json

  kubectl -n monitoring port-forward svc/grafana 3000:3000      # http://localhost:3000 (admin)
  kubectl -n monitoring port-forward svc/prometheus 9090:9090   # http://localhost:9090/targets
  Grafana password: kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.password}' | base64 -d

  Traffic for the dashboard: python scripts/load_test.py --url http://<EXTERNAL-IP>
MSG
