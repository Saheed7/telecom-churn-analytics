"""Model training module: train, tune, and compare ML classifiers."""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)

MODEL_DIR = Path("models")

# Model configurations with hyperparameter grids
MODEL_CONFIGS = {
    "LogisticRegression": {
        "model": LogisticRegression(max_iter=1000, random_state=42),
        "params": {
            "C": [0.01, 0.1, 1.0, 10.0],
            "penalty": ["l2"],
            "solver": ["lbfgs"],
        },
    },
    "RandomForest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5],
        },
    },
    "GradientBoosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
        },
    },
    "XGBoost": {
        "model": XGBClassifier(
            random_state=42, eval_metric="logloss", use_label_encoder=False
        ),
        "params": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1],
            "subsample": [0.8, 1.0],
        },
    },
}


def split_data(
    df: pd.DataFrame,
    target: str = "Churn",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split data into train/test sets with stratification.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered dataframe with target column.
    target : str
        Name of the target column.
    test_size : float
        Proportion of data to hold out for testing.
    random_state : int
        Random seed for reproducibility.

    Returns
    -------
    tuple
        X_train, X_test, y_train, y_test
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(
        "Train/test split: train=%d, test=%d, churn_rate_train=%.2f%%, churn_rate_test=%.2f%%",
        len(X_train), len(X_test),
        y_train.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_test, y_train, y_test


def apply_smote(
    X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """Apply SMOTE to handle class imbalance in the training set.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    random_state : int
        Random seed.

    Returns
    -------
    tuple
        Resampled X_train, y_train.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    logger.info(
        "SMOTE applied: %d -> %d samples (class dist: %s)",
        len(X_train), len(X_resampled),
        dict(pd.Series(y_resampled).value_counts()),
    )
    return pd.DataFrame(X_resampled, columns=X_train.columns), pd.Series(y_resampled)


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    models: Dict | None = None,
    cv_folds: int = 5,
) -> Dict:
    """Train and tune multiple models using GridSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    models : dict, optional
        Model configurations. Uses MODEL_CONFIGS by default.
    cv_folds : int
        Number of cross-validation folds.

    Returns
    -------
    dict
        Dictionary mapping model names to fitted GridSearchCV objects.
    """
    if models is None:
        models = MODEL_CONFIGS

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    trained_models = {}

    for name, config in models.items():
        logger.info("Training %s with GridSearchCV (%d folds)...", name, cv_folds)

        grid_search = GridSearchCV(
            estimator=config["model"],
            param_grid=config["params"],
            cv=cv,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=0,
        )
        grid_search.fit(X_train, y_train)
        trained_models[name] = grid_search

        logger.info(
            "%s — Best AUC: %.4f, Best Params: %s",
            name, grid_search.best_score_, grid_search.best_params_,
        )

    return trained_models


def select_best_model(trained_models: Dict) -> Tuple[str, object]:
    """Select the best performing model based on cross-validation AUC.

    Parameters
    ----------
    trained_models : dict
        Dictionary of trained GridSearchCV objects.

    Returns
    -------
    tuple
        (best_model_name, best_estimator)
    """
    best_name = max(trained_models, key=lambda k: trained_models[k].best_score_)
    best_model = trained_models[best_name].best_estimator_

    logger.info("Best model: %s (CV AUC: %.4f)",
                best_name, trained_models[best_name].best_score_)
    return best_name, best_model


def save_model(model: object, filepath: str | Path):
    """Serialize and save a trained model.

    Parameters
    ----------
    model : object
        Trained sklearn/xgboost model.
    filepath : str or Path
        Destination file path.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    logger.info("Model saved to %s", filepath)


def load_model(filepath: str | Path) -> object:
    """Load a serialized model from disk.

    Parameters
    ----------
    filepath : str or Path
        Path to the saved model file.

    Returns
    -------
    object
        Loaded model.
    """
    model = joblib.load(filepath)
    logger.info("Model loaded from %s", filepath)
    return model
