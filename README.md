# Heart Disease Risk — End-to-End MLOps

Binary classifier for heart-disease presence on the UCI Heart Disease (Cleveland) dataset,
taken from raw data to a monitored API on Kubernetes.

**Repository:** `https://github.com/<your-username>/heart-mlops` &nbsp;|&nbsp;
**Course:** AIMLCZG523 MLOps — Assignment 1

![architecture](reports/figures/architecture.png)

| Stage | Implementation |
|---|---|
| Data | `python -m heart.data` — official UCI file, GitHub mirror fallback; schema/range validation; target binarised (`num > 0`) |
| EDA | `python -m heart.eda` — class balance, per-class histograms, disease-rate by category, correlation heatmap, box plots |
| Features | One `sklearn` Pipeline: median impute + scale (numeric), mode impute (binary), mode impute + one-hot (categorical) |
| Models | Logistic Regression, Random Forest, HistGradientBoosting — `GridSearchCV`, 5-fold stratified, refit on ROC-AUC |
| Tracking | MLflow: 3 parent runs + 30 nested candidate runs; params, CV + test metrics, plots, model with input/output signature |
| Packaging | `models/model.joblib` (full pipeline) + `model_meta.json` (run id, metrics, data SHA-256, versions); MLflow model (skops) |
| CI/CD | GitHub Actions: ruff → pytest → train + quality gate → Docker build + smoke test → Trivy scan → push to GHCR |
| Serving | FastAPI: `/predict`, `/predict/batch`, `/health`, `/ready`, `/model`, `/metrics`; strict input schema |
| Deploy | Kubernetes (minikube): Deployment (2–5 pods, HPA, PDB, restricted PSS), LoadBalancer Service, nginx Ingress |
| Monitoring | Prometheus (pod discovery + alert rules) + provisioned Grafana dashboard; JSON access logs with request id |

## Results (held-out test set, n = 61)

| Model | CV ROC-AUC | Test ROC-AUC | Test accuracy | Test recall | Test precision |
|---|---|---|---|---|---|
| **Logistic Regression (champion)** | **0.907** | **0.960** | 0.869 | 0.929 | 0.813 |
| Random Forest | 0.901 | 0.955 | 0.885 | 0.929 | 0.839 |
| HistGradientBoosting | 0.889 | 0.951 | 0.885 | 0.893 | 0.862 |

The champion is chosen on **cross-validated** ROC-AUC, never on the test set.

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt -e .

python -m heart.data      # download + clean   -> data/processed/heart_clean.csv
python -m heart.eda       # figures            -> reports/figures/
python -m heart.train     # tune + track       -> mlflow.db, models/model.joblib
pytest -v && ruff check . # tests + lint
make mlflow               # MLflow UI on http://127.0.0.1:5000
make serve                # API on http://localhost:8000/docs
```

### Docker (Task 6)

```bash
./scripts/docker_smoke_test.sh     # build, run, /ready, /predict, 422 check, health status
```

### Kubernetes + monitoring (Tasks 7–8)

```bash
./scripts/deploy_minikube.sh       # builds image, loads it into minikube, applies k8s/ and k8s/monitoring
minikube tunnel                    # separate terminal: LoadBalancer external IP
kubectl -n heart get svc heart-api
curl -X POST http://<EXTERNAL-IP>/predict -H 'Content-Type: application/json' -d @sample_request.json
python scripts/load_test.py --url http://<EXTERNAL-IP>    # generate dashboard traffic
kubectl -n monitoring port-forward svc/grafana 3000:3000   # dashboard "Heart Disease API"
```

Local alternative without a cluster: `export GRAFANA_ADMIN_PASSWORD=$(openssl rand -hex 12) && docker compose up -d --build`.

### API

```bash
curl -X POST http://localhost:8000/predict -H 'Content-Type: application/json' -d @sample_request.json
# {"prediction":1,"label":"heart_disease","confidence":0.8986,"probability_disease":0.8986,
#  "model_name":"logistic_regression","model_version":"1.0.0"}
```

`ca` and `thal` are optional (imputed). Out-of-range values, invalid codes and unknown
fields return **422**; batches above 100 return **413**.

## Repository layout

```
src/heart/      config, data, eda, features, train, api
tests/          test_data.py, test_model.py, test_api.py (19 tests)
notebooks/      heart_disease_eda_modelling.ipynb (executed)
data/processed/ heart_clean.csv            models/  model.joblib, model_meta.json
k8s/            namespace, deployment, service, ingress, hpa+pdb, prometheus RBAC
k8s/monitoring/ prometheus + grafana (kustomize), dashboard JSON, alert rules
.github/workflows/ci.yml   Dockerfile   docker-compose.yml   Makefile
scripts/        docker_smoke_test.sh, deploy_minikube.sh, load_test.py, predict_client.py
reports/        figures, model_comparison.csv, eda_summary.json, report
screenshots/    evidence for the report
```

## Design notes

- **No train/serve skew:** the artifact is the whole fitted Pipeline; the API never re-implements preprocessing.
- **No leakage:** imputation/scaling statistics are learned inside each CV fold.
- **Traceability:** every image carries `model_meta.json` (MLflow run id, data SHA-256, library versions), exposed on `/model` and as the `model_info` metric.
- **Privacy:** logs record request id, route, status, latency and predicted label — never patient features.
- **Hardening:** non-root UID 10001, read-only root FS, all capabilities dropped, seccomp RuntimeDefault, restricted Pod Security Standard, minimal serving dependencies, Trivy gate in CI.
