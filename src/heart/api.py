"""FastAPI model-serving app: /predict, /predict/batch, /health, /ready, /metrics.

Design notes
- The persisted artifact is the *whole* sklearn Pipeline, so preprocessing at inference
  is byte-for-byte the code used in training (no train/serve skew).
- Strict input schema (ranges + allowed codes, unknown fields rejected) -> 422, never 500.
- Structured JSON access logs with a request id; raw patient features are NOT logged
  (health data), only the prediction outcome and latency.
- Prometheus metrics: request count/latency by route+status, prediction class counts,
  probability distribution, model info.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, ConfigDict, Field

from heart import config

MODEL_PATH = os.getenv("MODEL_PATH", str(config.MODEL_PATH))
META_PATH = os.getenv("MODEL_META_PATH", str(config.META_PATH))
MAX_BATCH = int(os.getenv("MAX_BATCH", "100"))


# ---------- logging ----------
class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(getattr(record, "extra_fields", {}))
        return json.dumps(payload)


log = logging.getLogger("heart.api")
if not log.handlers:
    _h = logging.StreamHandler(sys.stdout)
    _h.setFormatter(JsonFormatter())
    log.addHandler(_h)
    log.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    log.propagate = False

# ---------- metrics ----------
REQUESTS = Counter("http_requests_total", "HTTP requests", ["method", "route", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency",
    ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5),
)
PREDICTIONS = Counter("model_predictions_total", "Predictions by class", ["label"])
PROBABILITY = Histogram(
    "model_disease_probability",
    "Predicted P(disease)",
    buckets=[i / 10 for i in range(1, 11)],
)
MODEL_INFO = Gauge("model_info", "Loaded model", ["name", "version", "run_id"])


# ---------- schema ----------
class PatientFeatures(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={
            "example": {
                "age": 57,
                "sex": 1,
                "cp": 4,
                "trestbps": 140,
                "chol": 241,
                "fbs": 0,
                "restecg": 0,
                "thalach": 123,
                "exang": 1,
                "oldpeak": 0.2,
                "slope": 2,
                "ca": 0,
                "thal": 7,
            }
        },
    )

    age: int = Field(ge=18, le=100, description="Age in years")
    sex: Literal[0, 1] = Field(description="1 = male, 0 = female")
    cp: Literal[1, 2, 3, 4] = Field(description="Chest pain type (4 = asymptomatic)")
    trestbps: float = Field(ge=70, le=250, description="Resting blood pressure (mm Hg)")
    chol: float = Field(ge=80, le=650, description="Serum cholesterol (mg/dl)")
    fbs: Literal[0, 1] = Field(description="Fasting blood sugar > 120 mg/dl")
    restecg: Literal[0, 1, 2] = Field(description="Resting ECG result")
    thalach: float = Field(ge=50, le=230, description="Max heart rate achieved")
    exang: Literal[0, 1] = Field(description="Exercise-induced angina")
    oldpeak: float = Field(ge=0, le=10, description="ST depression induced by exercise")
    slope: Literal[1, 2, 3] = Field(description="Slope of peak exercise ST segment")
    ca: int | None = Field(default=None, ge=0, le=3, description="Major vessels (0-3)")
    thal: Literal[3, 6, 7] | None = Field(default=None, description="3 normal/6 fixed/7 rev.")


class Prediction(BaseModel):
    prediction: int
    label: str
    confidence: float = Field(description="Probability of the predicted class")
    probability_disease: float
    model_name: str
    model_version: str


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    instances: list[PatientFeatures] = Field(min_length=1)


# ---------- model ----------
class ModelHolder:
    pipeline = None
    meta: dict = {}
    threshold: float = 0.5

    @classmethod
    def load(cls) -> None:
        cls.pipeline = joblib.load(MODEL_PATH)
        cls.meta = json.loads(open(META_PATH).read())
        cls.threshold = float(cls.meta.get("decision_threshold", 0.5))
        MODEL_INFO.labels(
            cls.meta["model_name"], cls.meta["model_version"], cls.meta.get("mlflow_run_id", "")
        ).set(1)
        log.info(
            "model loaded",
            extra={
                "extra_fields": {
                    "model": cls.meta["model_name"],
                    "version": cls.meta["model_version"],
                }
            },
        )


def _predict(rows: list[PatientFeatures]) -> list[Prediction]:
    if ModelHolder.pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    X = pd.DataFrame([r.model_dump() for r in rows], columns=config.FEATURES).astype(float)
    probs = ModelHolder.pipeline.predict_proba(X)[:, 1]
    out = []
    for p in probs:
        pred = int(p >= ModelHolder.threshold)
        label = "heart_disease" if pred else "no_heart_disease"
        PREDICTIONS.labels(label).inc()
        PROBABILITY.observe(float(p))
        out.append(
            Prediction(
                prediction=pred,
                label=label,
                confidence=round(float(p if pred else 1 - p), 4),
                probability_disease=round(float(p), 4),
                model_name=ModelHolder.meta["model_name"],
                model_version=ModelHolder.meta["model_version"],
            )
        )
    return out


@asynccontextmanager
async def lifespan(_: FastAPI):
    ModelHolder.load()
    yield


app = FastAPI(title="Heart Disease Risk API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def access_log(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    route = request.url.path
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        response.headers["x-request-id"] = request_id
        return response
    finally:
        elapsed = time.perf_counter() - start
        if route != "/metrics":
            REQUESTS.labels(request.method, route, str(status)).inc()
            LATENCY.labels(route).observe(elapsed)
            log.info(
                "request",
                extra={
                    "extra_fields": {
                        "request_id": request_id,
                        "method": request.method,
                        "path": route,
                        "status": status,
                        "latency_ms": round(elapsed * 1000, 2),
                        "client": request.client.host if request.client else None,
                    }
                },
            )


@app.get("/health")
def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
def ready():
    """Readiness: the model is loaded and can serve."""
    if ModelHolder.pipeline is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {
        "status": "ready",
        "model": ModelHolder.meta.get("model_name"),
        "version": ModelHolder.meta.get("model_version"),
    }


@app.get("/model")
def model_info():
    return ModelHolder.meta


@app.post("/predict", response_model=Prediction)
def predict(features: PatientFeatures):
    result = _predict([features])[0]
    log.info(
        "prediction",
        extra={
            "extra_fields": {
                "label": result.label,
                "probability_disease": result.probability_disease,
            }
        },
    )
    return result


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(req: BatchRequest):
    if len(req.instances) > MAX_BATCH:
        raise HTTPException(status_code=413, detail=f"max {MAX_BATCH} instances per request")
    return _predict(req.instances)


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
