"""Exploratory Data Analysis module: visualization and statistical tests."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

logger = logging.getLogger(__name__)

# Set consistent style
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
FIGSIZE = (10, 6)
DPI = 150
FIG_DIR = Path("reports/figures")


def _save_fig(fig: plt.Figure, name: str, output_dir: Optional[Path] = None):
    """Save figure to the reports directory."""
    save_dir = output_dir or FIG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    filepath = save_dir / f"{name}.png"
    fig.savefig(filepath, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("Saved figure: %s", filepath)


def plot_churn_distribution(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Plot target variable (Churn) distribution.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'Churn' column.
    output_dir : Path, optional
        Directory to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Count plot
    churn_counts = df["Churn"].value_counts()
    colors = ["#2ecc71", "#e74c3c"]
    axes[0].bar(["No Churn", "Churn"], churn_counts.values, color=colors, edgecolor="white")
    axes[0].set_title("Churn Distribution (Count)", fontweight="bold")
    axes[0].set_ylabel("Number of Customers")

    for i, v in enumerate(churn_counts.values):
        axes[0].text(i, v + 50, f"{v:,}", ha="center", fontweight="bold")

    # Pie chart
    axes[1].pie(churn_counts.values, labels=["No Churn", "Churn"],
                autopct="%1.1f%%", colors=colors, startangle=90,
                explode=(0, 0.05), shadow=True)
    axes[1].set_title("Churn Distribution (%)", fontweight="bold")

    fig.suptitle("Customer Churn Overview", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "churn_distribution", output_dir)


def plot_correlation_heatmap(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Plot correlation heatmap for numeric features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numeric columns.
    output_dir : Path, optional
        Directory to save the figure.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()

    fig, ax = plt.subplots(figsize=(12, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, square=True, linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, "correlation_heatmap", output_dir)


def plot_numeric_distributions(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Plot distributions of key numeric features by churn status.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with Churn column.
    output_dir : Path, optional
        Directory to save the figure.
    """
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, col in zip(axes, numeric_features):
        for label, color in [(0, "#2ecc71"), (1, "#e74c3c")]:
            subset = df[df["Churn"] == label][col]
            ax.hist(subset, bins=30, alpha=0.6, color=color,
                    label="No Churn" if label == 0 else "Churn", edgecolor="white")
        ax.set_title(f"{col} Distribution by Churn", fontweight="bold")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.suptitle("Numeric Feature Distributions", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "numeric_distributions", output_dir)


def plot_categorical_churn_rates(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Plot churn rates across categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with categorical and Churn columns.
    output_dir : Path, optional
        Directory to save the figure.
    """
    cat_features = ["Contract", "InternetService", "PaymentMethod"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for ax, col in zip(axes, cat_features):
        churn_rate = df.groupby(col)["Churn"].mean().sort_values(ascending=False)
        bars = ax.bar(range(len(churn_rate)), churn_rate.values,
                      color=sns.color_palette("viridis", len(churn_rate)),
                      edgecolor="white")
        ax.set_xticks(range(len(churn_rate)))
        ax.set_xticklabels(churn_rate.index, rotation=30, ha="right")
        ax.set_title(f"Churn Rate by {col}", fontweight="bold")
        ax.set_ylabel("Churn Rate")
        ax.set_ylim(0, 1)

        for bar, val in zip(bars, churn_rate.values):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                    f"{val:.1%}", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle("Churn Rate by Categorical Features", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "categorical_churn_rates", output_dir)


def plot_boxplots(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Plot box plots of charges by churn status.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    output_dir : Path, optional
        Directory to save the figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, col in zip(axes, ["MonthlyCharges", "TotalCharges"]):
        churn_labels = df["Churn"].map({0: "No Churn", 1: "Churn"})
        sns.boxplot(x=churn_labels, y=df[col], ax=ax,
                    palette=["#2ecc71", "#e74c3c"], width=0.5)
        ax.set_title(f"{col} by Churn Status", fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel(col)

    fig.suptitle("Charge Distributions by Churn", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    _save_fig(fig, "charge_boxplots", output_dir)


def chi_square_tests(df: pd.DataFrame) -> pd.DataFrame:
    """Perform chi-square tests of independence for categoricals vs Churn.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with categorical features and Churn column.

    Returns
    -------
    pd.DataFrame
        Results with chi2 statistic, p-value, and significance flag.
    """
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    results = []

    for col in cat_cols:
        contingency = pd.crosstab(df[col], df["Churn"])
        chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
        results.append({
            "Feature": col,
            "Chi2": round(chi2, 4),
            "P-Value": round(p_value, 6),
            "DoF": dof,
            "Significant (p<0.05)": p_value < 0.05,
        })
        logger.info("Chi-square test — %s: chi2=%.4f, p=%.6f", col, chi2, p_value)

    return pd.DataFrame(results).sort_values("P-Value")


def run_eda(df: pd.DataFrame, output_dir: Optional[Path] = None):
    """Run the full EDA pipeline and generate all visualizations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    output_dir : Path, optional
        Directory to save all figures.
    """
    logger.info("Starting EDA pipeline...")
    plot_churn_distribution(df, output_dir)
    plot_correlation_heatmap(df, output_dir)
    plot_numeric_distributions(df, output_dir)
    plot_categorical_churn_rates(df, output_dir)
    plot_boxplots(df, output_dir)

    chi_results = chi_square_tests(df)
    logger.info("Chi-square test results:\n%s", chi_results.to_string())

    logger.info("EDA pipeline complete.")
    return chi_results
