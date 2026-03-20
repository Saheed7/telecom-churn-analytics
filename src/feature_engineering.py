"""Feature engineering module: create and transform predictive features."""

import logging
from typing import Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def create_tenure_bins(df: pd.DataFrame) -> pd.DataFrame:
    """Bin tenure into meaningful customer lifecycle segments.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with 'tenure' column (in months).

    Returns
    -------
    pd.DataFrame
        Dataframe with 'tenure_bin' column added.
    """
    df = df.copy()
    bins = [0, 12, 24, 48, 72]
    labels = ["0-12m", "12-24m", "24-48m", "48-72m"]
    df["tenure_bin"] = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    logger.info("Created tenure bins: %s", df["tenure_bin"].value_counts().to_dict())
    return df


def create_charge_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer charge-related features.

    Creates:
    - AvgMonthlyCharge: TotalCharges / tenure (average monthly spend)
    - ChargeRatio: MonthlyCharges / TotalCharges (recency weight)

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with tenure, MonthlyCharges, TotalCharges.

    Returns
    -------
    pd.DataFrame
        Dataframe with new charge features.
    """
    df = df.copy()

    # Average monthly charge (handle zero tenure)
    df["AvgMonthlyCharge"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    # Charge ratio (handle zero TotalCharges)
    df["ChargeRatio"] = np.where(
        df["TotalCharges"] > 0,
        df["MonthlyCharges"] / df["TotalCharges"],
        0,
    )

    logger.info("Created charge features: AvgMonthlyCharge, ChargeRatio")
    return df


def create_service_count(df: pd.DataFrame) -> pd.DataFrame:
    """Count the number of active services per customer.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with service columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with 'NumServices' and 'ChargePerService' columns.
    """
    df = df.copy()

    # Count services where value is 'Yes' (string) or 1 (after encoding)
    service_cols_present = [c for c in SERVICE_COLUMNS if c in df.columns]

    def count_services(row):
        count = 0
        for col in service_cols_present:
            val = row[col]
            if val == 1 or val == "Yes":
                count += 1
        return count

    df["NumServices"] = df.apply(count_services, axis=1)
    df["ChargePerService"] = np.where(
        df["NumServices"] > 0,
        df["MonthlyCharges"] / df["NumServices"],
        df["MonthlyCharges"],
    )

    logger.info("Created service features: NumServices (mean=%.2f), ChargePerService",
                df["NumServices"].mean())
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode remaining multi-class categorical features.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with categorical columns to encode.

    Returns
    -------
    pd.DataFrame
        Dataframe with one-hot encoded features.
    """
    df = df.copy()
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True, dtype=int)
        logger.info("One-hot encoded %d categorical columns: %s", len(cat_cols), cat_cols)

    return df


def scale_numeric_features(df: pd.DataFrame,
                            exclude: list | None = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """Standardize numeric features using StandardScaler.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with numeric features.
    exclude : list, optional
        Column names to exclude from scaling (e.g., target variable).

    Returns
    -------
    tuple of (pd.DataFrame, StandardScaler)
        Scaled dataframe and fitted scaler object.
    """
    df = df.copy()
    exclude = exclude or []
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c not in exclude]

    scaler = StandardScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    logger.info("Scaled %d numeric features", len(numeric_cols))
    return df, scaler


def engineer_features(df: pd.DataFrame, scale: bool = True) -> Tuple[pd.DataFrame, StandardScaler | None]:
    """Run the full feature engineering pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe.
    scale : bool
        Whether to apply standard scaling to numeric features.

    Returns
    -------
    tuple of (pd.DataFrame, StandardScaler or None)
        Engineered dataframe and optional scaler.
    """
    logger.info("Starting feature engineering pipeline...")

    df = create_tenure_bins(df)
    df = create_charge_features(df)
    df = create_service_count(df)
    df = encode_categorical_features(df)

    scaler = None
    if scale:
        df, scaler = scale_numeric_features(df, exclude=["Churn"])

    logger.info("Feature engineering complete. Shape: %s", df.shape)
    return df, scaler
