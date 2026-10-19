"""Exploratory data analysis: writes figures + a summary JSON to reports/."""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

from heart import config  # noqa: E402
from heart.data import load_clean  # noqa: E402

PALETTE = {0: "#2a9d8f", 1: "#e76f51"}
LABELS = {0: "No disease", 1: "Disease"}


def _save(fig, name: str) -> None:
    config.FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIG_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)


def class_balance(df: pd.DataFrame) -> None:
    counts = df[config.TARGET].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    bars = ax.bar(
        [LABELS[i] for i in counts.index], counts.values, color=[PALETTE[i] for i in counts.index]
    )
    for b, v in zip(bars, counts.values, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v} ({v / len(df):.0%})", ha="center")
    ax.set_title("Class balance")
    ax.set_ylabel("Patients")
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, "01_class_balance.png")


def numeric_histograms(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, col in zip(axes.flat, config.NUMERIC, strict=True):
        for cls in (0, 1):
            sns.histplot(
                df.loc[df[config.TARGET] == cls, col].dropna(),
                ax=ax,
                kde=True,
                color=PALETTE[cls],
                label=LABELS[cls],
                stat="density",
                alpha=0.45,
                bins=20,
            )
        ax.set_title(col)
        ax.set_ylabel("")
    axes.flat[0].legend()
    fig.suptitle("Numeric feature distributions by class", y=1.01)
    fig.tight_layout()
    _save(fig, "02_numeric_histograms.png")


def categorical_rates(df: pd.DataFrame) -> None:
    cols = config.BINARY + config.CATEGORICAL
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    for ax, col in zip(axes.flat, cols, strict=False):
        rate = df.groupby(col)[config.TARGET].agg(["mean", "size"]).reset_index()
        ax.bar(rate[col].astype(int).astype(str), rate["mean"], color="#264653")
        for i, (m, n) in enumerate(zip(rate["mean"], rate["size"], strict=True)):
            ax.text(i, m + 0.02, f"n={n}", ha="center", fontsize=8)
        ax.set_ylim(0, 1.1)
        ax.set_title(f"{col}: disease rate")
    axes.flat[-1].axis("off")
    fig.tight_layout()
    _save(fig, "03_categorical_disease_rate.png")


def correlation_heatmap(df: pd.DataFrame) -> pd.Series:
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        ax=ax,
        square=True,
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 7},
    )
    ax.set_title("Pearson correlation (incl. target)")
    _save(fig, "04_correlation_heatmap.png")
    return corr[config.TARGET].drop(config.TARGET).sort_values(key=abs, ascending=False)


def outlier_boxplots(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 6, figsize=(16, 3.8))
    for ax, col in zip(axes, config.NUMERIC, strict=True):
        sns.boxplot(
            data=df, x=config.TARGET, y=col, hue=config.TARGET, ax=ax, palette=PALETTE, legend=False
        )
        ax.set_xlabel("target")
    fig.tight_layout()
    _save(fig, "05_boxplots_by_class.png")


def main() -> None:
    df = load_clean()
    class_balance(df)
    numeric_histograms(df)
    categorical_rates(df)
    target_corr = correlation_heatmap(df)
    outlier_boxplots(df)
    summary = {
        "rows": len(df),
        "missing_by_column": {k: int(v) for k, v in df.isna().sum().items() if v},
        "positive_rate": round(float(df[config.TARGET].mean()), 4),
        "target_correlation": {k: round(float(v), 3) for k, v in target_corr.items()},
        "describe": df.describe().round(2).to_dict(),
    }
    (config.ROOT / "reports" / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in list(summary)[:4]}, indent=2))


if __name__ == "__main__":
    main()
