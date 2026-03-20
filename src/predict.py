"""Prediction module: inference on new customer data."""

import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger(__name__)


class ChurnPredictor:
    """Wrapper for making churn predictions on new customer data.

    Parameters
    ----------
    model_path : str or Path
        Path to the serialized trained model.
    scaler_path : str or Path, optional
        Path to the fitted StandardScaler. If None, no scaling is applied.

    Examples
    --------
    >>> predictor = ChurnPredictor("models/best_model.joblib")
    >>> prob = predictor.predict_single({"tenure": 24, "MonthlyCharges": 79.85, ...})
    >>> print(f"Churn probability: {prob:.2%}")
    """

    def __init__(self, model_path: str | Path, scaler_path: str | Path | None = None):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path) if scaler_path else None
        self.feature_names = None

        # Try to extract feature names from the model
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)

        logger.info("ChurnPredictor initialized with model from %s", model_path)

    def predict_single(self, customer_data: Dict) -> float:
        """Predict churn probability for a single customer.

        Parameters
        ----------
        customer_data : dict
            Dictionary of feature name -> value pairs.

        Returns
        -------
        float
            Churn probability (0.0 to 1.0).
        """
        df = pd.DataFrame([customer_data])
        probabilities = self.predict_batch(df)
        return probabilities[0]

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Predict churn probabilities for a batch of customers.

        Parameters
        ----------
        df : pd.DataFrame
            Dataframe of customer features.

        Returns
        -------
        np.ndarray
            Array of churn probabilities.
        """
        # Align columns with model expectations
        if self.feature_names:
            missing = set(self.feature_names) - set(df.columns)
            for col in missing:
                df[col] = 0
            df = df[self.feature_names]

        # Apply scaling if scaler is available
        if self.scaler is not None:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = self.scaler.transform(df[numeric_cols])

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(df)[:, 1]
        return self.model.predict(df).astype(float)

    def predict_with_explanation(self, customer_data: Dict) -> Dict:
        """Predict with top contributing features.

        Parameters
        ----------
        customer_data : dict
            Customer feature dictionary.

        Returns
        -------
        dict
            Prediction result with probability and risk level.
        """
        probability = self.predict_single(customer_data)

        if probability >= 0.7:
            risk_level = "HIGH"
        elif probability >= 0.4:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return {
            "churn_probability": round(probability, 4),
            "risk_level": risk_level,
            "recommendation": _get_recommendation(risk_level),
        }


def _get_recommendation(risk_level: str) -> str:
    """Return retention recommendation based on risk level."""
    recommendations = {
        "HIGH": "Immediate intervention recommended: offer loyalty discount, "
                "personal account review, or contract upgrade incentive.",
        "MEDIUM": "Proactive engagement suggested: send satisfaction survey, "
                  "highlight unused benefits, or offer service bundle.",
        "LOW": "Continue standard engagement: periodic check-ins and "
               "new feature announcements.",
    }
    return recommendations.get(risk_level, "Monitor customer activity.")
