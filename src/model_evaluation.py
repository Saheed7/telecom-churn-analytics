"""Model evaluation module: metrics, plots, SHAP interpretability."""

import logging
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve,
    classification_report, confusion_matrix,
)

logger = logging.getLogger(__name__)

FIG_DIR = Path("reports/figures")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: Optional[np.ndarray] = None) -> dict:
    """Compute classification metrics.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for positive class.

    Returns
    -------
    dict
        Dictionary of evaluation metrics.
    """
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)

    return {k: round(v, 4) for k, v in metrics.items()}


def evaluate_all_models(
    trained_models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> pd.DataFrame:
    """Evaluate all trained models and return a comparison table.

    Parameters
    ----------
    trained_models : dict
        Dictionary mapping model names to fitted GridSearchCV objects.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.

    Returns
    -------
    pd.DataFrame
        Comparison table of model metrics.
    """
    results = []
    for name, grid in trained_models.items():
        model = grid.best_estimator_
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = compute_metrics(y_test, y_pred, y_prob)
        metrics["model"] = name
        metrics["cv_auc"] = round(grid.best_score_, 4)
        results.append(metrics)

        logger.info("%s — %s", name, metrics)

    df_results = pd.DataFrame(results).set_index("model")
    df_results = df_results.sort_values("roc_auc", ascending=False)
    return df_results


def plot_roc_curves(
    trained_models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[Path] = None,
):
    """Plot ROC curves for all models.

    Parameters
    ----------
    trained_models : dict
        Trained model dictionary.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    output_dir : Path, optional
        Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, grid in trained_models.items():
        model = grid.best_estimator_
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc = roc_auc_score(y_test, y_prob)
            ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Baseline")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(alpha=0.3)

    save_dir = output_dir or FIG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved ROC curves plot")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    output_dir: Optional[Path] = None,
):
    """Plot a confusion matrix heatmap.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    model_name : str
        Name of the model (for the title).
    output_dir : Path, optional
        Directory to save the figure.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Churn", "Churn"],
                yticklabels=["No Churn", "Churn"])
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")

    save_dir = output_dir or FIG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f"confusion_matrix_{model_name.lower()}.png",
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved confusion matrix for %s", model_name)


def plot_precision_recall_curves(
    trained_models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[Path] = None,
):
    """Plot precision-recall curves for all models.

    Parameters
    ----------
    trained_models : dict
        Trained model dictionary.
    X_test, y_test : array-like
        Test data.
    output_dir : Path, optional
        Save directory.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    for name, grid in trained_models.items():
        model = grid.best_estimator_
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            ax.plot(recall, precision, label=name, linewidth=2)

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(alpha=0.3)

    save_dir = output_dir or FIG_DIR
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / "precision_recall_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved precision-recall curves plot")


def plot_shap_summary(model, X_test: pd.DataFrame, output_dir: Optional[Path] = None):
    """Generate SHAP summary plot for model interpretability.

    Parameters
    ----------
    model : object
        Trained model (tree-based recommended).
    X_test : pd.DataFrame
        Test features.
    output_dir : Path, optional
        Save directory.
    """
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False, max_display=15)
        plt.title("SHAP Feature Importance (Top 15)", fontsize=14, fontweight="bold")
        plt.tight_layout()

        save_dir = output_dir or FIG_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved SHAP summary plot")

    except ImportError:
        logger.warning("SHAP not installed. Skipping interpretability plot.")
    except Exception as e:
        logger.warning("SHAP analysis failed: %s", e)


def generate_evaluation_report(
    trained_models: Dict,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Generate full evaluation report with plots.

    Parameters
    ----------
    trained_models : dict
        Trained model dictionary.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Test labels.
    output_dir : Path, optional
        Save directory.

    Returns
    -------
    pd.DataFrame
        Model comparison results table.
    """
    logger.info("Generating evaluation report...")

    results = evaluate_all_models(trained_models, X_test, y_test)
    plot_roc_curves(trained_models, X_test, y_test, output_dir)
    plot_precision_recall_curves(trained_models, X_test, y_test, output_dir)

    # Confusion matrix for best model
    best_name = results.index[0]
    best_model = trained_models[best_name].best_estimator_
    y_pred = best_model.predict(X_test)
    plot_confusion_matrix(y_test, y_pred, best_name, output_dir)

    # SHAP for best model
    plot_shap_summary(best_model, X_test, output_dir)

    logger.info("Evaluation report complete.")
    return results
