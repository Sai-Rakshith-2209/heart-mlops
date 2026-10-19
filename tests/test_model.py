import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score

from heart import config
from heart.features import build_pipeline, build_preprocessor
from heart.train import model_space


def test_preprocessor_output_is_finite_and_expected_width(clean_df):
    Xt = build_preprocessor().fit_transform(clean_df[config.FEATURES])
    # 6 numeric + 3 binary + one-hot(cp 4, restecg 3, slope 3, thal 3)
    assert Xt.shape == (len(clean_df), 6 + 3 + 13)
    assert np.isfinite(Xt).all()


def test_preprocessor_tolerates_missing_and_unseen_values(clean_df):
    pre = build_preprocessor().fit(clean_df[config.FEATURES])
    row = clean_df[config.FEATURES].head(1).copy()
    row["ca"] = np.nan
    row["thal"] = 99  # unseen category -> all-zero one-hot, no error
    assert np.isfinite(pre.transform(row)).all()


def test_pipeline_beats_baseline_in_cv(clean_df):
    pipe = build_pipeline(LogisticRegression(max_iter=2000))
    auc = cross_val_score(
        pipe, clean_df[config.FEATURES], clean_df[config.TARGET], cv=5, scoring="roc_auc"
    ).mean()
    assert auc > 0.85


def test_pipeline_is_deterministic(clean_df):
    X, y = clean_df[config.FEATURES], clean_df[config.TARGET]
    p1 = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y).predict_proba(X)
    p2 = build_pipeline(LogisticRegression(max_iter=2000)).fit(X, y).predict_proba(X)
    np.testing.assert_allclose(p1, p2)


@pytest.mark.parametrize("name", list(model_space()))
def test_every_model_family_trains_and_predicts(name, clean_df):
    est, _ = model_space()[name]
    pipe = build_pipeline(est).fit(clean_df[config.FEATURES], clean_df[config.TARGET])
    prob = pipe.predict_proba(clean_df[config.FEATURES])[:, 1]
    assert ((prob >= 0) & (prob <= 1)).all()
    assert roc_auc_score(clean_df[config.TARGET], prob) > 0.8
