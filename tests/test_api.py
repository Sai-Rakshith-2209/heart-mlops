def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready")
    assert r.status_code == 200 and r.json()["model"] == "test_lr"


def test_predict_returns_prediction_and_confidence(client, sample_patient):
    r = client.post("/predict", json=sample_patient)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.5 <= body["confidence"] <= 1.0
    assert 0.0 <= body["probability_disease"] <= 1.0
    assert body["label"] in ("heart_disease", "no_heart_disease")
    assert "x-request-id" in r.headers


def test_predict_accepts_missing_optional_fields(client, sample_patient):
    sample_patient.pop("ca")
    sample_patient.pop("thal")
    assert client.post("/predict", json=sample_patient).status_code == 200


def test_predict_rejects_out_of_range(client, sample_patient):
    sample_patient["age"] = 250
    assert client.post("/predict", json=sample_patient).status_code == 422


def test_predict_rejects_invalid_code_and_unknown_field(client, sample_patient):
    bad = {**sample_patient, "cp": 7}
    assert client.post("/predict", json=bad).status_code == 422
    extra = {**sample_patient, "ssn": "123"}
    assert client.post("/predict", json=extra).status_code == 422


def test_batch_predict_and_limit(client, sample_patient):
    r = client.post("/predict/batch", json={"instances": [sample_patient] * 3})
    assert r.status_code == 200 and len(r.json()) == 3
    r = client.post("/predict/batch", json={"instances": [sample_patient] * 101})
    assert r.status_code == 413


def test_metrics_exposed(client, sample_patient):
    client.post("/predict", json=sample_patient)
    text = client.get("/metrics").text
    assert "http_requests_total" in text
    assert "model_predictions_total" in text
    assert 'model_info{name="test_lr"' in text


def test_metric_labels_are_bounded(client):
    """Unknown paths must collapse to one label, not one time series per URL."""
    for path in ("/wp-admin", "/.env", "/random-123"):
        assert client.get(path).status_code == 404
    text = client.get("/metrics").text
    assert 'route="unmatched"' in text
    assert "/wp-admin" not in text and "/random-123" not in text


def test_request_id_is_validated(client, sample_patient):
    ok = client.post("/predict", json=sample_patient, headers={"x-request-id": "abc-123"})
    assert ok.headers["x-request-id"] == "abc-123"
    bad = client.post("/predict", json=sample_patient, headers={"x-request-id": "x" * 500})
    assert bad.headers["x-request-id"] != "x" * 500
    assert len(bad.headers["x-request-id"]) == 32  # freshly minted uuid4 hex
