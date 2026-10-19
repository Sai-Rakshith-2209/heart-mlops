# Screenshots for the report

| File | What to capture | Command |
|---|---|---|
| 01_mlflow_runs.png | MLflow experiment with 3 parent + 30 nested runs | `make mlflow` |
| 02_mlflow_run_detail.png | One run: metrics + plots artifacts | MLflow UI |
| 03_github_actions.png | All 4 CI jobs green + artifacts | Actions tab |
| 04_ci_failure.png | Pipeline failing on a broken test | push a failing test on a branch |
| 05_docker_smoke.png | Build, /ready, /predict, 422, healthy | `./scripts/docker_smoke_test.sh` |
| 06_k8s_resources.png | Pods, Service EXTERNAL-IP, Ingress, HPA | `kubectl get all,ingress,hpa -n heart` |
| 07_k8s_predict.png | curl via LoadBalancer and Ingress | see README |
| 08_prometheus_targets.png | Both pods UP | port-forward 9090 |
| 09_grafana_dashboard.png | Dashboard with traffic | `python scripts/load_test.py` |
| 10_pod_logs.png | JSON request logs | `kubectl -n heart logs deploy/heart-api` |
