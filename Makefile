.PHONY: setup data eda train test lint mlflow serve docker compose k8s load clean

setup:            ## create venv and install everything
	python3 -m venv .venv && .venv/bin/pip install -r requirements-dev.txt -e .
data:             ## download + clean the dataset
	python -m heart.data
eda:
	python -m heart.eda
train:            ## tune 3 model families, log to MLflow, save champion
	python -m heart.train
lint:
	ruff check . && ruff format --check .
test:
	pytest -v
mlflow:           ## MLflow UI on http://127.0.0.1:5000
	mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000
serve:
	uvicorn heart.api:app --reload --port 8000
docker:
	./scripts/docker_smoke_test.sh
compose:
	docker compose up -d --build
k8s:
	./scripts/deploy_minikube.sh
load:
	python scripts/load_test.py --url http://localhost:8000
clean:
	rm -rf mlruns mlflow.db .pytest_cache .ruff_cache
