"""Train, tune and compare classifiers with full MLflow tracking; package the champion.

Flow
  1. stratified 80/20 train/test split (test set is touched once, for reporting only)
  2. per model family: GridSearchCV (5-fold stratified CV, refit on ROC-AUC)
     - one MLflow parent run per family, one nested child run per grid point
  3. champion = best *CV* ROC-AUC (never selected on the test set)
  4. champion pipeline (preprocessing + model) saved as joblib + MLflow model
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
import skops.io as sio  # noqa: E402
from mlflow.models import infer_signature  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split  # noqa: E402

from heart import __version__, config  # noqa: E402
from heart.data import file_sha256, load_clean  # noqa: E402
from heart.features import build_pipeline  # noqa: E402

os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
log = logging.getLogger("heart.train")
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]
EXPERIMENT = "heart-disease-classifier"


def model_space() -> dict[str, tuple[object, dict]]:
    rs = config.RANDOM_STATE
    return {
        "logistic_regression": (
            LogisticRegression(max_iter=5000, solver="liblinear", random_state=rs),
            {"model__C": [0.01, 0.1, 0.3, 1.0, 3.0], "model__penalty": ["l1", "l2"]},
        ),
        "random_forest": (
            RandomForestClassifier(random_state=rs, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 4, 8],
                "model__min_samples_leaf": [1, 5],
            },
        ),
        "hist_gradient_boosting": (
            HistGradientBoostingClassifier(random_state=rs),
            {
                "model__learning_rate": [0.03, 0.1],
                "model__max_depth": [2, 3],
                "model__max_iter": [100, 200],
            },
        ),
    }


def test_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "test_accuracy": accuracy_score(y_true, y_pred),
        "test_precision": precision_score(y_true, y_pred, zero_division=0),
        "test_recall": recall_score(y_true, y_pred),
        "test_f1": f1_score(y_true, y_pred),
        "test_roc_auc": roc_auc_score(y_true, y_prob),
    }


def feature_importance(pipe, X_test, y_test) -> pd.Series:
    """Native importances where available, else permutation importance on raw features."""
    model = pipe.named_steps["model"]
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=names)
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=names)
    from sklearn.inspection import permutation_importance

    r = permutation_importance(
        pipe, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=config.RANDOM_STATE
    )
    return pd.Series(r.importances_mean, index=X_test.columns)


def save_plots(name, pipe, X_test, y_test, out_dir) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ConfusionMatrixDisplay.from_estimator(
        pipe, X_test, y_test, ax=ax, cmap="Blues", display_labels=["No disease", "Disease"]
    )
    ax.set_title(f"{name}: confusion matrix (test)")
    paths.append(out_dir / f"{name}_confusion_matrix.png")
    fig.savefig(paths[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
    RocCurveDisplay.from_estimator(pipe, X_test, y_test, ax=axes[0])
    axes[0].plot([0, 1], [0, 1], "--", color="grey")
    axes[0].set_title("ROC (test)")
    PrecisionRecallDisplay.from_estimator(pipe, X_test, y_test, ax=axes[1])
    axes[1].set_title("Precision-Recall (test)")
    fig.suptitle(name)
    paths.append(out_dir / f"{name}_roc_pr.png")
    fig.savefig(paths[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)

    imp = feature_importance(pipe, X_test, y_test).sort_values().tail(12)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.barh(imp.index, imp.values, color="#264653")
    ax.set_title(f"{name}: top feature importance")
    paths.append(out_dir / f"{name}_feature_importance.png")
    fig.savefig(paths[-1], dpi=130, bbox_inches="tight")
    plt.close(fig)
    return [str(p) for p in paths]


def comparison_plot(results: pd.DataFrame, out_path) -> None:
    cols = ["cv_roc_auc", "test_roc_auc", "test_accuracy", "test_recall", "test_precision"]
    ax = results.set_index("model")[cols].T.plot(kind="bar", figsize=(9, 4), rot=0, width=0.8)
    ax.set_ylim(0.6, 1.0)
    ax.set_title("Model comparison")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.figure.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(ax.figure)


def train(tracking_uri: str, n_jobs: int = -1) -> dict:
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(EXPERIMENT)

    df = load_clean()
    X, y = df[config.FEATURES], df[config.TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=config.RANDOM_STATE)
    data_hash = file_sha256(config.CLEAN_PATH)
    plot_dir = config.FIG_DIR / "models"

    rows, fitted = [], {}
    for name, (estimator, grid) in model_space().items():
        log.info("Tuning %s over %d candidates", name, len(list(_grid_iter(grid))))
        search = GridSearchCV(
            build_pipeline(estimator),
            grid,
            cv=cv,
            scoring=SCORING,
            refit="roc_auc",
            n_jobs=n_jobs,
            return_train_score=True,
        )
        with mlflow.start_run(run_name=name) as parent:
            mlflow.set_tags(
                {
                    "model_family": name,
                    "data_sha256": data_hash,
                    "stage": "tuning",
                    "git_sha": os.getenv("GITHUB_SHA", "local"),
                }
            )
            mlflow.log_params(
                {
                    "cv_folds": 5,
                    "test_size": config.TEST_SIZE,
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                    "random_state": config.RANDOM_STATE,
                }
            )
            search.fit(X_train, y_train)

            # One nested run per grid point -> every candidate is tracked.
            res = search.cv_results_
            for i, params in enumerate(res["params"]):
                with mlflow.start_run(run_name=f"{name}-cand{i:02d}", nested=True):
                    mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                    mlflow.log_metrics(
                        {
                            **{f"cv_{m}": float(res[f"mean_test_{m}"][i]) for m in SCORING},
                            **{f"cv_{m}_std": float(res[f"std_test_{m}"][i]) for m in SCORING},
                            "cv_train_roc_auc": float(res["mean_train_roc_auc"][i]),
                        }
                    )

            best = search.best_estimator_
            y_prob = best.predict_proba(X_test)[:, 1]
            y_pred = (y_prob >= 0.5).astype(int)
            bi = search.best_index_
            cv_m = {f"cv_{m}": float(res[f"mean_test_{m}"][bi]) for m in SCORING}
            tm = test_metrics(y_test, y_pred, y_prob)

            mlflow.log_params(
                {f"best_{k.replace('model__', '')}": v for k, v in search.best_params_.items()}
            )
            mlflow.log_metrics({**cv_m, **tm, "cv_roc_auc_std": float(res["std_test_roc_auc"][bi])})
            for p in save_plots(name, best, X_test, y_test, plot_dir):
                mlflow.log_artifact(p, artifact_path="plots")
            mlflow.sklearn.log_model(
                best,
                name="model",
                signature=infer_signature(X_test, best.predict_proba(X_test)),
                input_example=X_test.head(3),
                # skops (not pickle) serialisation. We built this object in-process, so we
                # allow-list exactly the types it contains - nothing broader.
                skops_trusted_types=sio.get_untrusted_types(data=sio.dumps(best)),
            )
            rows.append(
                {
                    "model": name,
                    "run_id": parent.info.run_id,
                    "best_params": {
                        k.replace("model__", ""): v for k, v in search.best_params_.items()
                    },
                    **cv_m,
                    **{k: float(v) for k, v in tm.items()},
                }
            )
            fitted[name] = best
            log.info("%s  cv_auc=%.4f  test_auc=%.4f", name, cv_m["cv_roc_auc"], tm["test_roc_auc"])

    results = pd.DataFrame(rows).sort_values("cv_roc_auc", ascending=False)
    champion = results.iloc[0]
    comparison_plot(results, config.FIG_DIR / "model_comparison.png")
    results.drop(columns=["best_params"]).to_csv(
        config.ROOT / "reports" / "model_comparison.csv", index=False
    )

    config.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(fitted[champion["model"]], config.MODEL_PATH)
    meta = {
        "model_name": champion["model"],
        "model_version": __version__,
        "mlflow_run_id": champion["run_id"],
        "best_params": champion["best_params"],
        "metrics": {
            k: round(float(champion[k]), 4)
            for k in champion.index
            if k.startswith(("cv_", "test_"))
        },
        "decision_threshold": 0.5,
        "features": config.FEATURES,
        "data_sha256": data_hash,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    config.META_PATH.write_text(json.dumps(meta, indent=2, default=str))

    with mlflow.start_run(run_id=champion["run_id"]):
        mlflow.set_tag("champion", "true")
        mlflow.log_artifact(str(config.META_PATH))
        mlflow.log_artifact(str(config.FIG_DIR / "model_comparison.png"), artifact_path="plots")
    log.info(
        "Champion: %s (cv_roc_auc=%.4f) -> %s",
        champion["model"],
        champion["cv_roc_auc"],
        config.MODEL_PATH,
    )
    return meta


def _grid_iter(grid):
    from sklearn.model_selection import ParameterGrid

    return ParameterGrid(grid)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument(
        "--tracking-uri",
        default=os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{config.ROOT}/mlflow.db"),
    )
    p.add_argument("--n-jobs", type=int, default=-1)
    args = p.parse_args()
    meta = train(args.tracking_uri, args.n_jobs)
    print(json.dumps(meta["metrics"], indent=2))


if __name__ == "__main__":
    main()
