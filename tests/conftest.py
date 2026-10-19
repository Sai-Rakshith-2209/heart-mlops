"""Shared fixtures. API tests use a small model trained on the fly, so the test job
does not depend on the (slower) training job's artifacts."""

import json
import os

import joblib
import pytest
from sklearn.linear_model import LogisticRegression

from heart import config
from heart.data import load_clean
from heart.features import build_pipeline


@pytest.fixture(scope="session")
def clean_df():
    return load_clean()


@pytest.fixture(scope="session")
def model_files(tmp_path_factory, clean_df):
    d = tmp_path_factory.mktemp("model")
    pipe = build_pipeline(LogisticRegression(max_iter=2000))
    pipe.fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    joblib.dump(pipe, d / "model.joblib")
    (d / "model_meta.json").write_text(
        json.dumps(
            {
                "model_name": "test_lr",
                "model_version": "0.0.0-test",
                "mlflow_run_id": "test",
                "decision_threshold": 0.5,
            }
        )
    )
    return d


@pytest.fixture(scope="session")
def client(model_files):
    os.environ["MODEL_PATH"] = str(model_files / "model.joblib")
    os.environ["MODEL_META_PATH"] = str(model_files / "model_meta.json")
    from fastapi.testclient import TestClient

    from heart import api

    api.MODEL_PATH = os.environ["MODEL_PATH"]
    api.META_PATH = os.environ["MODEL_META_PATH"]
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def sample_patient():
    return {
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
