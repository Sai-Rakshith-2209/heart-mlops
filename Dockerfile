# syntax=docker/dockerfile:1
# Serving image. The model artifact is produced by `python -m heart.train` (locally or in
# CI) and copied in, so the image contains exactly the MLflow-tracked champion model
# (its run id is in models/model_meta.json and exposed on /model and /metrics).

FROM python:3.12-slim AS deps
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /build
COPY requirements-serve.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install -r requirements-serve.txt

FROM python:3.12-slim AS runtime
LABEL org.opencontainers.image.title="heart-disease-api" \
      org.opencontainers.image.description="Heart disease risk classifier (FastAPI)"
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    MODEL_PATH=/app/models/model.joblib \
    MODEL_META_PATH=/app/models/model_meta.json
# Unprivileged user with a fixed UID so Kubernetes runAsNonRoot can verify it.
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --no-create-home app
WORKDIR /app
COPY --from=deps /opt/venv /opt/venv
COPY src/heart/__init__.py src/heart/config.py src/heart/api.py ./src/heart/
COPY models/model.joblib models/model_meta.json ./models/
USER 10001
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2).status==200 else 1)"
CMD ["uvicorn", "heart.api:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
