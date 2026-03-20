"""Data cleaning module: handle missing values, type casting, and outliers."""

import logging
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def cast_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """Convert TotalCharges from string to numeric, handling whitespace.

    The raw dataset contains whitespace strings in TotalCharges for new
    customers with zero tenure. These are converted to NaN for imputation.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with TotalCharges as object/string type.

    Returns
    -------
    pd.DataFrame
        Dataframe with TotalCharges as float64.
    """
    df = df.copy()
    df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    n_converted = df["TotalCharges"].isna().sum()
    logger.info("TotalCharges: %d values converted to NaN", n_converted)
    return df


def impute_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    """Impute missing numeric values using the specified strategy.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    strategy : str
        Imputation strategy: 'median', 'mean', or 'zero'.

    Returns
    -------
    pd.DataFrame
        Dataframe with missing numeric values imputed.
    """
    df = df.copy()
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            if strategy == "median":
                fill_value = df[col].median()
            elif strategy == "mean":
                fill_value = df[col].mean()
            elif strategy == "zero":
                fill_value = 0
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            df[col] = df[col].fillna(fill_value)
            logger.info("Imputed %d missing values in '%s' with %s=%.2f",
                        n_missing, col, strategy, fill_value)

    return df


def encode_binary_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode binary categorical features as 0/1 integers.

    Converts Yes/No columns and the target variable (Churn) to numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.

    Returns
    -------
    pd.DataFrame
        Dataframe with binary features encoded.
    """
    df = df.copy()
    binary_map = {"Yes": 1, "No": 0}

    binary_cols = ["Partner", "Dependents", "PhoneService",
                   "PaperlessBilling", "Churn"]

    for col in binary_cols:
        if col in df.columns and df[col].dtype == object:
            df[col] = df[col].map(binary_map)
            logger.info("Encoded binary column: %s", col)

    # Gender encoding
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": 1, "Female": 0})
        logger.info("Encoded gender column")

    return df


def detect_outliers_iqr(df: pd.DataFrame, columns: Optional[list] = None,
                         factor: float = 1.5) -> pd.DataFrame:
    """Detect outliers using the IQR method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    columns : list, optional
        Columns to check. If None, checks all numeric columns.
    factor : float
        IQR multiplier for determining outlier bounds (default 1.5).

    Returns
    -------
    pd.DataFrame
        Boolean dataframe where True indicates an outlier.
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = pd.DataFrame(False, index=df.index, columns=columns)

    for col in columns:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        lower = q1 - factor * iqr
        upper = q3 + factor * iqr
        outlier_mask[col] = (df[col] < lower) | (df[col] > upper)
        n_outliers = outlier_mask[col].sum()
        if n_outliers > 0:
            logger.info("Column '%s': %d outliers detected (bounds: %.2f, %.2f)",
                        col, n_outliers, lower, upper)

    return outlier_mask


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full cleaning pipeline.

    Steps:
    1. Cast TotalCharges to numeric
    2. Impute missing values (median)
    3. Encode binary features
    4. Drop customerID (non-predictive)

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe.

    Returns
    -------
    pd.DataFrame
        Cleaned dataframe ready for feature engineering.
    """
    logger.info("Starting data cleaning pipeline...")
    df = cast_total_charges(df)
    df = impute_missing_values(df, strategy="median")
    df = encode_binary_features(df)

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
        logger.info("Dropped customerID column")

    logger.info("Cleaning complete. Shape: %s", df.shape)
    return df
