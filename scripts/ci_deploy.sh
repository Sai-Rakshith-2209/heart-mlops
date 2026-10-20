#!/usr/bin/env bash
# Deploy the API + monitoring stack to minikube and collect evidence (Tasks 7 & 8).
# Runs on a GitHub Actions runner or any Linux machine with Docker, kubectl and minikube.
# Expects the image heart-disease-api:1.0.0 to exist in the local Docker daemon.
set -euo pipefail
cd "$(dirname "$0")/.."
IMAGE=heart-disease-api:1.0.0
OUT=deploy-evidence
mkdir -p "$OUT"

log() { printf '\n=== %s\n' "$*"; }

log "Start minikube"
minikube start --driver=docker --cpus=2 --memory=6g --wait=all
minikube addons enable ingress
minikube addons enable metrics-server
minikube image load "$IMAGE"

log "Deploy API"
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/
kubectl -n heart rollout status deploy/heart-api --timeout=300s

log "Deploy monitoring"
kubectl -n monitoring get secret grafana-admin >/dev/null 2>&1 ||
  kubectl -n monitoring create secret generic grafana-admin \
    --from-literal=password="$(openssl rand -hex 12)"
kubectl apply -k k8s/monitoring
kubectl -n monitoring rollout status deploy/prometheus --timeout=300s
kubectl -n monitoring rollout status deploy/grafana --timeout=300s
kubectl -n ingress-nginx wait --for=condition=ready pod \
  -l app.kubernetes.io/component=controller --timeout=300s

log "Expose the LoadBalancer (minikube tunnel assigns the EXTERNAL-IP)"
nohup minikube tunnel >"$OUT/tunnel.log" 2>&1 &
LB_IP=""
for _ in $(seq 1 45); do
  LB_IP=$(kubectl -n heart get svc heart-api -o jsonpath='{.status.loadBalancer.ingress[0].ip}')
  [ -n "$LB_IP" ] && break
  sleep 2
done
if [ -n "$LB_IP" ] && curl -sf -m 5 "http://$LB_IP/ready" >/dev/null; then
  API_URL="http://$LB_IP"
  VIA="LoadBalancer EXTERNAL-IP"
else
  # Every LoadBalancer Service also gets a NodePort; use it if the tunnel is unavailable.
  NODE_PORT=$(kubectl -n heart get svc heart-api -o jsonpath='{.spec.ports[0].nodePort}')
  API_URL="http://$(minikube ip):$NODE_PORT"
  VIA="Service NodePort (tunnel unavailable)"
fi
MK_IP=$(minikube ip)
echo "API reachable at $API_URL via $VIA"

log "Verify endpoints"
{
  echo "\$ kubectl get deploy,pods,svc,ingress,hpa,pdb -n heart -o wide"
  kubectl get deploy,pods,svc,ingress,hpa,pdb -n heart -o wide
  echo
  echo "\$ kubectl get pods -n monitoring"
  kubectl get pods -n monitoring
  echo
  echo "# API reachable at $API_URL via $VIA"
  echo "\$ curl $API_URL/ready"
  curl -sf "$API_URL/ready"
  echo
  echo
  echo "\$ curl -X POST $API_URL/predict -d @sample_request.json"
  curl -sf -X POST "$API_URL/predict" -H 'Content-Type: application/json' -d @sample_request.json
  echo
  echo
  echo "\$ curl -X POST -H 'Host: heart.local' http://$MK_IP/predict -d @sample_request.json   # via Ingress"
  for _ in $(seq 1 30); do
    curl -sf -o /dev/null -H 'Host: heart.local' "http://$MK_IP/ready" && break
    sleep 2
  done
  curl -sf -X POST "http://$MK_IP/predict" -H 'Host: heart.local' \
    -H 'Content-Type: application/json' -d @sample_request.json
  echo
  echo
  echo "\$ curl -X POST $API_URL/predict -d '{\"age\": 999}'   # invalid input"
  curl -s -o /dev/null -w 'HTTP %{http_code}\n' -X POST "$API_URL/predict" \
    -H 'Content-Type: application/json' -d '{"age": 999}'
} 2>&1 | tee "$OUT/k8s_deploy.txt"

log "Port-forward Prometheus, Grafana and the API docs"
nohup kubectl -n monitoring port-forward svc/prometheus 9090:9090 >"$OUT/pf-prom.log" 2>&1 &
nohup kubectl -n monitoring port-forward svc/grafana 3000:3000 >"$OUT/pf-grafana.log" 2>&1 &
nohup kubectl -n heart port-forward svc/heart-api 8080:80 >"$OUT/pf-api.log" 2>&1 &
for port in 9090 3000 8080; do
  for _ in $(seq 1 30); do curl -s -o /dev/null "localhost:$port" && break; sleep 1; done
done

log "Generate traffic for the dashboard"
python scripts/load_test.py --url "$API_URL" --requests 400 --rps 20 | tee "$OUT/load_test.txt"
sleep 45 # let Prometheus scrape (10s interval) and the rate() windows fill

log "Collect logs and Prometheus state"
{
  echo "\$ kubectl -n heart logs deploy/heart-api --tail=15"
  kubectl -n heart logs deploy/heart-api --tail=15
} | tee "$OUT/pod_logs.txt"
curl -s localhost:9090/api/v1/targets | python -c "
import json, sys
for t in json.load(sys.stdin)['data']['activeTargets']:
    print(t['labels'].get('pod'), t['scrapeUrl'], t['health'])
" | tee "$OUT/prometheus_targets.txt"
curl -s 'localhost:9090/api/v1/query?query=sum%20by%20(route,status)(http_requests_total)' |
  python -m json.tool >"$OUT/prometheus_query.json"

log "Capture screenshots"
GRAFANA_PASSWORD=$(kubectl -n monitoring get secret grafana-admin -o jsonpath='{.data.password}' | base64 -d)
GRAFANA_PASSWORD="$GRAFANA_PASSWORD" python scripts/capture_evidence.py --out "$OUT"
ls -la "$OUT"