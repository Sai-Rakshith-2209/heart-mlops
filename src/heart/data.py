"""Data acquisition and cleaning for the UCI Heart Disease (Cleveland) dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import ssl
import urllib.request

import certifi
import numpy as np
import pandas as pd

from heart import config

log = logging.getLogger(__name__)

# ISLR mirror encodes some categoricals as strings; map back to UCI codebook values.
_MIRROR_MAPS = {
    "ChestPain": {"typical": 1, "nontypical": 2, "nonanginal": 3, "asymptomatic": 4},
    "Thal": {"normal": 3, "fixed": 6, "reversable": 7},
    "AHD": {"No": 0, "Yes": 1},
}
_MIRROR_RENAME = {
    "Age": "age",
    "Sex": "sex",
    "ChestPain": "cp",
    "RestBP": "trestbps",
    "Chol": "chol",
    "Fbs": "fbs",
    "RestECG": "restecg",
    "MaxHR": "thalach",
    "ExAng": "exang",
    "Oldpeak": "oldpeak",
    "Slope": "slope",
    "Ca": "ca",
    "Thal": "thal",
    "AHD": "num",
}


# One explicit, pinned CA bundle (Mozilla roots via certifi) on every machine, instead of
# whatever the OS store happens to contain. Verification stays fully enabled.
_TLS_CONTEXT = ssl.create_default_context(cafile=certifi.where())


def _fetch(url: str, timeout: int = 30) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout, context=_TLS_CONTEXT) as resp:  # noqa: S310
        return resp.read()


def parse_uci(raw: bytes) -> pd.DataFrame:
    """Parse the headerless UCI processed.cleveland.data file ('?' = missing)."""
    return pd.read_csv(io.BytesIO(raw), header=None, names=config.COLUMNS, na_values="?")


def parse_mirror(raw: bytes) -> pd.DataFrame:
    """Parse the ISLR mirror and convert it to the UCI schema."""
    df = pd.read_csv(io.BytesIO(raw), index_col=0, na_values=["NA", "?"])
    for col, mapping in _MIRROR_MAPS.items():
        df[col] = df[col].map(mapping)
    return df.rename(columns=_MIRROR_RENAME)[config.COLUMNS].reset_index(drop=True)


def download(dest=config.RAW_PATH) -> pd.DataFrame:
    """Download the dataset (UCI first, mirror fallback) and save it in UCI schema."""
    try:
        df = parse_uci(_fetch(config.UCI_URL))
        source = config.UCI_URL
    except Exception as exc:  # network / host blocked
        log.warning("UCI download failed (%s); using mirror", exc)
        df = parse_mirror(_fetch(config.MIRROR_URL))
        source = config.MIRROR_URL
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False, lineterminator="\n")
    log.info("Saved %d rows from %s to %s", len(df), source, dest)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce types, null out impossible values, drop duplicates, binarise the target.

    Missing feature values are kept as NaN on purpose: they are imputed inside the
    model pipeline so imputation statistics are learned on training folds only
    (no leakage) and the same logic runs at inference time.
    """
    out = df.copy()
    for col in config.COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    for col, allowed in config.CATEGORY_DOMAINS.items():
        out.loc[~out[col].isin(allowed), col] = np.nan
    for col, (lo, hi) in config.NUMERIC_RANGES.items():
        out.loc[(out[col] < lo) | (out[col] > hi), col] = np.nan

    out = out.dropna(subset=["num"]).drop_duplicates().reset_index(drop=True)
    # num: 0 = no disease, 1-4 = increasing severity -> binary presence/absence.
    out[config.TARGET] = (out["num"] > 0).astype(int)
    return out.drop(columns=["num"])


def load_clean(path=config.CLEAN_PATH) -> pd.DataFrame:
    return pd.read_csv(path)


def file_sha256(path) -> str:
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Download and clean the heart disease data")
    p.add_argument("--skip-download", action="store_true", help="re-clean existing raw file")
    args = p.parse_args()

    raw = pd.read_csv(config.RAW_PATH) if args.skip_download else download()
    cleaned = clean(raw)
    config.CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Force LF so the file (and its SHA-256) is byte-identical on Windows and Linux.
    cleaned.to_csv(config.CLEAN_PATH, index=False, lineterminator="\n")
    log.info(
        "Clean data: %d rows, %d missing cells, positive rate %.3f -> %s",
        len(cleaned),
        int(cleaned.isna().sum().sum()),
        cleaned[config.TARGET].mean(),
        config.CLEAN_PATH,
    )


if __name__ == "__main__":
    main()
