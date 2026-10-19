"""Render reports/figures/architecture.png (used in README and report)."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

from heart import config  # noqa: E402

C = {"data": "#e9f5f3", "ml": "#fdf0e9", "ci": "#eef1f8", "run": "#f3eef8", "obs": "#fbf6e6"}
EDGE = "#33415c"


def box(ax, x, y, w, h, title, lines, color):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=color, ec=EDGE, lw=1.2
        )
    )
    ax.text(
        x + w / 2,
        y + h - 0.22,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
        weight="bold",
        color=EDGE,
    )
    ax.text(
        x + w / 2,
        y + h - 0.58,
        "\n".join(lines),
        ha="center",
        va="top",
        fontsize=8.3,
        color="#222",
        linespacing=1.45,
    )


def arrow(ax, a, b, label=None, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            a,
            b,
            arrowstyle="-|>",
            mutation_scale=13,
            lw=1.3,
            color=EDGE,
            connectionstyle=f"arc3,rad={rad}",
        )
    )
    if label:
        ax.text(
            (a[0] + b[0]) / 2,
            (a[1] + b[1]) / 2 + 0.12,
            label,
            ha="center",
            fontsize=7.8,
            color="#555",
            style="italic",
        )


def group(ax, x, y, w, h, label):
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.2",
            fc="none",
            ec="#9aa5b8",
            lw=1,
            ls="--",
        )
    )
    ax.text(x + 0.15, y + h - 0.08, label, fontsize=9, color="#6b7487", va="top", weight="bold")


fig, ax = plt.subplots(figsize=(15, 8.2))
ax.set_xlim(0, 15)
ax.set_ylim(0, 8.2)
ax.axis("off")

group(ax, 0.2, 4.3, 9.4, 3.75, "Experimentation & training (local / CI runner)")
box(
    ax,
    0.4,
    5.5,
    2.6,
    2.1,
    "1. Data",
    ["heart.data", "UCI download", "(mirror fallback)", "clean + validate"],
    C["data"],
)
box(
    ax,
    3.4,
    5.5,
    2.6,
    2.1,
    "2. EDA",
    ["heart.eda", "histograms, heatmap", "class balance", "disease rates"],
    C["data"],
)
box(
    ax,
    6.4,
    5.5,
    3.0,
    2.1,
    "3. Train + tune",
    ["sklearn Pipeline", "(impute/scale/one-hot", "+ model) x 3 families", "GridSearchCV 5-fold"],
    C["ml"],
)
box(ax, 6.4, 4.45, 3.0, 0.9, "MLflow tracking", [], C["ml"])
ax.text(7.9, 4.68, "33 runs: params, metrics, plots, models", ha="center", fontsize=8)
box(ax, 0.4, 4.45, 5.6, 0.9, "Artifacts", [], C["data"])
ax.text(3.2, 4.68, "heart_clean.csv | model.joblib + model_meta.json", ha="center", fontsize=8)

group(ax, 9.9, 4.3, 4.9, 3.75, "GitHub Actions CI/CD")
box(
    ax,
    10.1,
    5.5,
    4.5,
    2.1,
    "Pipeline",
    [
        "lint (ruff)  ->  unit tests (pytest)",
        "->  train + quality gate (AUC/recall)",
        "->  docker build + smoke test",
        "->  Trivy scan  ->  push GHCR (main)",
    ],
    C["ci"],
)
box(ax, 10.1, 4.45, 4.5, 0.9, "Run artifacts", [], C["ci"])
ax.text(12.35, 4.68, "junit, model, mlflow.db, reports, logs", ha="center", fontsize=8)

group(ax, 0.2, 0.2, 14.6, 3.85, "Kubernetes (minikube / Docker Desktop)")
box(ax, 0.5, 1.2, 2.4, 1.9, "Client", ["curl / app", "JSON POST", "/predict"], C["run"])
box(
    ax,
    3.4,
    1.2,
    2.6,
    1.9,
    "Ingress / LB",
    ["nginx ingress", "heart.local", "Service type", "LoadBalancer"],
    C["run"],
)
box(
    ax,
    6.5,
    0.6,
    3.6,
    3.1,
    "heart-api Deployment",
    [
        "FastAPI + uvicorn, 2-5 pods (HPA)",
        "non-root, read-only FS, no caps",
        "/predict  /predict/batch",
        "/health  /ready  /model",
        "/metrics (Prometheus)",
        "JSON logs + request id",
    ],
    C["run"],
)
box(ax, 10.6, 2.15, 1.9, 1.5, "Prometheus", ["pod discovery", "alert rules"], C["obs"])
box(ax, 12.8, 2.15, 1.8, 1.5, "Grafana", ["provisioned", "dashboard"], C["obs"])
box(ax, 10.6, 0.5, 4.0, 1.25, "kubectl logs", ["structured access logs (no PHI)"], C["obs"])

arrow(ax, (3.0, 6.55), (3.4, 6.55))
arrow(ax, (6.0, 6.55), (6.4, 6.55))
arrow(ax, (7.9, 5.5), (7.9, 5.35))
arrow(ax, (6.4, 4.9), (6.0, 4.9))
arrow(ax, (9.4, 6.55), (10.1, 6.55), "git push")
arrow(ax, (12.35, 4.45), (9.2, 3.7), rad=-0.12)
ax.text(11.35, 3.95, "image (GHCR / minikube load)", fontsize=7.8, color="#555", style="italic")
arrow(ax, (2.9, 2.15), (3.4, 2.15))
arrow(ax, (6.0, 2.15), (6.5, 2.15))
arrow(ax, (10.6, 2.9), (10.1, 2.9), "scrape")
arrow(ax, (12.5, 2.9), (12.8, 2.9))
arrow(ax, (10.1, 1.1), (10.6, 1.1))

fig.savefig(config.FIG_DIR / "architecture.png", dpi=150, bbox_inches="tight")
print("written", config.FIG_DIR / "architecture.png")
