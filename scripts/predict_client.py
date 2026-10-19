"""Inference examples: (1) in-process with the saved pipeline, (2) over HTTP.

Usage: python scripts/predict_client.py [--url http://localhost:8000]
"""

import argparse
import json
import urllib.request

import joblib
import pandas as pd

from heart import config

SAMPLE = json.load(open(config.ROOT / "sample_request.json"))


def local() -> None:
    pipe = joblib.load(config.MODEL_PATH)
    proba = pipe.predict_proba(pd.DataFrame([SAMPLE])[config.FEATURES])[0, 1]
    print(f"local pipeline  -> P(disease) = {proba:.4f}")


def remote(url: str) -> None:
    req = urllib.request.Request(
        f"{url}/predict",
        data=json.dumps(SAMPLE).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=5) as r:
        print(f"HTTP {url} -> {json.loads(r.read())}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url")
    args = ap.parse_args()
    local()
    if args.url:
        remote(args.url)
