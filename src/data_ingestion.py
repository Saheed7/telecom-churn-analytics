"""Data ingestion module: load, validate, and profile raw telecom data."""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

EXPECTED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
]


def load_data(filepath: str | Path) -> pd.DataFrame:
    """Load CSV data with basic validation.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataframe with validated schema.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If required columns are missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    logger.info("Loading data from %s", filepath)
    df = pd.read_csv(filepath)
    logger.info("Loaded %d rows, %d columns", *df.shape)

    # Validate schema
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    return df


def profile_data(df: pd.DataFrame) -> dict:
    """Generate a data quality profile report.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe to profile.

    Returns
    -------
    dict
        Dictionary containing profile statistics.
    """
    profile = {
        "shape": df.shape,
        "dtypes": df.dtypes.value_counts().to_dict(),
        "missing_counts": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().sum() / len(df) * 100).round(2).to_dict(),
        "duplicates": df.duplicated().sum(),
        "numeric_stats": df.describe().to_dict(),
    }

    total_missing = df.isnull().sum().sum()
    logger.info(
        "Profile: %d rows, %d cols, %d total missing values, %d duplicates",
        *df.shape, total_missing, profile["duplicates"],
    )
    return profile


def generate_quality_report(df: pd.DataFrame, output_path: Optional[str] = None) -> str:
    """Create a human-readable data quality report.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe.
    output_path : str, optional
        If provided, save the report to this path.

    Returns
    -------
    str
        Formatted quality report string.
    """
    profile = profile_data(df)
    lines = [
        "=" * 60,
        "DATA QUALITY REPORT",
        "=" * 60,
        f"Rows: {profile['shape'][0]:,}",
        f"Columns: {profile['shape'][1]}",
        f"Duplicate rows: {profile['duplicates']}",
        "",
        "MISSING VALUES:",
        "-" * 40,
    ]

    for col, count in profile["missing_counts"].items():
        if count > 0:
            pct = profile["missing_pct"][col]
            lines.append(f"  {col}: {count} ({pct}%)")

    if all(v == 0 for v in profile["missing_counts"].values()):
        lines.append("  None detected.")

    lines.extend(["", "COLUMN TYPES:", "-" * 40])
    for dtype, count in profile["dtypes"].items():
        lines.append(f"  {dtype}: {count} columns")

    report = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(report)
        logger.info("Quality report saved to %s", output_path)

    return report
