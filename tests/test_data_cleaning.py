"""Tests for the data cleaning module."""

import pytest
import pandas as pd
import numpy as np

from src.data_cleaning import (
    cast_total_charges,
    impute_missing_values,
    encode_binary_features,
    detect_outliers_iqr,
    clean_data,
)


@pytest.fixture
def sample_raw_df():
    """Create a sample raw dataframe mimicking the Telco dataset."""
    return pd.DataFrame({
        "customerID": ["C001", "C002", "C003", "C004", "C005"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No", "Yes"],
        "tenure": [1, 34, 2, 45, 0],
        "PhoneService": ["No", "Yes", "Yes", "No", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "No", "No internet service", "Yes"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service", "No"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service", "Yes"],
        "TechSupport": ["No", "No", "No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No", "Yes", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No", "Yes", "No internet service", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "One year"],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check", "Mailed check", "Electronic check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 20.05, 89.10],
        "TotalCharges": ["29.85", "1889.5", "108.15", "   ", "0"],
        "Churn": ["No", "No", "Yes", "No", "Yes"],
    })


class TestCastTotalCharges:
    """Tests for cast_total_charges function."""

    def test_converts_string_to_numeric(self, sample_raw_df):
        result = cast_total_charges(sample_raw_df)
        assert result["TotalCharges"].dtype == np.float64

    def test_whitespace_becomes_nan(self, sample_raw_df):
        result = cast_total_charges(sample_raw_df)
        # Row index 3 has whitespace in TotalCharges
        assert pd.isna(result.loc[3, "TotalCharges"])

    def test_valid_values_preserved(self, sample_raw_df):
        result = cast_total_charges(sample_raw_df)
        assert result.loc[0, "TotalCharges"] == 29.85
        assert result.loc[1, "TotalCharges"] == 1889.5

    def test_zero_string_converts(self, sample_raw_df):
        result = cast_total_charges(sample_raw_df)
        assert result.loc[4, "TotalCharges"] == 0.0


class TestImputeMissingValues:
    """Tests for impute_missing_values function."""

    def test_median_imputation(self):
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = impute_missing_values(df, strategy="median")
        assert result["A"].isna().sum() == 0
        assert result.loc[2, "A"] == 3.0  # median of [1, 2, 4, 5]

    def test_mean_imputation(self):
        df = pd.DataFrame({"A": [1.0, 2.0, np.nan, 4.0, 5.0]})
        result = impute_missing_values(df, strategy="mean")
        assert result.loc[2, "A"] == 3.0  # mean of [1, 2, 4, 5]

    def test_zero_imputation(self):
        df = pd.DataFrame({"A": [1.0, np.nan, 3.0]})
        result = impute_missing_values(df, strategy="zero")
        assert result.loc[1, "A"] == 0.0

    def test_no_missing_values(self):
        df = pd.DataFrame({"A": [1.0, 2.0, 3.0]})
        result = impute_missing_values(df, strategy="median")
        pd.testing.assert_frame_equal(result, df)

    def test_invalid_strategy_raises(self):
        df = pd.DataFrame({"A": [1.0, np.nan]})
        with pytest.raises(ValueError, match="Unknown strategy"):
            impute_missing_values(df, strategy="invalid")


class TestEncodeBinaryFeatures:
    """Tests for encode_binary_features function."""

    def test_encodes_yes_no_columns(self, sample_raw_df):
        result = encode_binary_features(sample_raw_df)
        assert result["Partner"].dtype in [np.int64, int]
        assert set(result["Partner"].unique()) <= {0, 1}

    def test_encodes_churn(self, sample_raw_df):
        result = encode_binary_features(sample_raw_df)
        assert result["Churn"].dtype in [np.int64, int]
        assert result.loc[0, "Churn"] == 0
        assert result.loc[2, "Churn"] == 1

    def test_encodes_gender(self, sample_raw_df):
        result = encode_binary_features(sample_raw_df)
        assert result["gender"].dtype in [np.int64, int]
        assert result.loc[0, "gender"] == 1  # Male
        assert result.loc[1, "gender"] == 0  # Female

    def test_does_not_modify_original(self, sample_raw_df):
        original_churn = sample_raw_df["Churn"].copy()
        encode_binary_features(sample_raw_df)
        pd.testing.assert_series_equal(sample_raw_df["Churn"], original_churn)


class TestDetectOutliersIQR:
    """Tests for detect_outliers_iqr function."""

    def test_detects_outliers(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5, 100]})
        mask = detect_outliers_iqr(df, columns=["A"])
        assert mask["A"].sum() >= 1  # 100 should be an outlier

    def test_no_outliers(self):
        df = pd.DataFrame({"A": [1, 2, 3, 4, 5]})
        mask = detect_outliers_iqr(df, columns=["A"])
        assert mask["A"].sum() == 0

    def test_returns_boolean_dataframe(self):
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        mask = detect_outliers_iqr(df)
        assert mask.dtypes["A"] == bool
        assert mask.dtypes["B"] == bool


class TestCleanData:
    """Tests for the full clean_data pipeline."""

    def test_drops_customer_id(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert "customerID" not in result.columns

    def test_total_charges_numeric(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert result["TotalCharges"].dtype == np.float64

    def test_no_missing_values_after_cleaning(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert result.isnull().sum().sum() == 0

    def test_binary_columns_encoded(self, sample_raw_df):
        result = clean_data(sample_raw_df)
        assert result["Churn"].dtype in [np.int64, int]
        assert result["Partner"].dtype in [np.int64, int]
