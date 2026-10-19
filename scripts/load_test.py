"""Send realistic + some invalid traffic to the API so dashboards have data.

Usage: python scripts/load_test.py --url http://localhost:8000 --requests 300
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request

import pandas as pd

FEATURES = [
    "age",
    "sex",
    "cp",
    "trestbps",
    "chol",
    "fbs",
    "restecg",
    "thalach",
    "exang",
    "oldpeak",
    "slope",
    "ca",
    "thal",
]


def post(url: str, body: dict) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status
    except urllib.error.HTTPError as e:
        return e.code


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://localhost:8000")
    p.add_argument("--requests", type=int, default=300)
    p.add_argument("--rps", type=float, default=10)
    a = p.parse_args()

    rows = pd.read_csv("data/processed/heart_clean.csv")[FEATURES].dropna()
    codes: dict[int, int] = {}
    for i in range(a.requests):
        row = {
            k: (int(v) if k not in ("oldpeak",) else float(v))
            for k, v in rows.sample(1).iloc[0].items()
        }
        if random.random() < 0.05:  # 5% malformed -> exercises 422 path
            row["age"] = 999
        code = post(f"{a.url}/predict", row)
        codes[code] = codes.get(code, 0) + 1
        time.sleep(1 / a.rps)
        if (i + 1) % 50 == 0:
            print(f"{i + 1} sent, status counts: {codes}")
    print("done", codes)


if __name__ == "__main__":
    main()
