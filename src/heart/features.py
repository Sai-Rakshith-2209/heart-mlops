"""Preprocessing pipeline shared by training and inference (single source of truth)."""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from heart import config


def build_preprocessor() -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    binary = SimpleImputer(strategy="most_frequent")
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            # handle_unknown="ignore": an unseen category at serving time becomes all-zeros
            # instead of crashing the request.
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric, config.NUMERIC),
            ("bin", binary, config.BINARY),
            ("cat", categorical, config.CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(estimator) -> Pipeline:
    return Pipeline([("preprocess", build_preprocessor()), ("model", estimator)])
