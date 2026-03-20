"""Tests for the feature engineering module."""

import pytest
import pandas as pd
import numpy as np

from src.feature_engineering import (
    create_tenure_bins,
    create_charge_features,
    create_service_count,
    encode_categorical_features,
    scale_numeric_features,
)


@pytest.fixture
def sample_cleaned_df():
    """Create a sample cleaned dataframe for feature engineering tests."""
    return pd.DataFrame({
        "gender": [1, 0, 1, 0, 1],
        "SeniorCitizen": [0, 1, 0, 0, 1],
        "Partner": [1, 0, 1, 0, 1],
        "Dependents": [0, 0, 1, 0, 1],
        "tenure": [1, 34, 2, 45, 60],
        "PhoneService": [0, 1, 1, 0, 1],
        "MultipleLines": ["No phone service", "No", "Yes", "No phone service", "No"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "No", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes", "No", "No internet service", "Yes"],
        "OnlineBackup": ["Yes", "No", "No", "No internet service", "No"],
        "DeviceProtection": ["No", "Yes", "No", "No internet service", "Yes"],
        "TechSupport": ["No", "No", "No", "No internet service", "Yes"],
        "StreamingTV": ["No", "No", "Yes", "No internet service", "Yes"],
        "StreamingMovies": ["No", "No", "Yes", "No internet service", "No"],
        "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year", "One year"],
        "PaperlessBilling": [1, 0, 1, 0, 1],
        "PaymentMethod": [
            "Electronic check", "Mailed check", "Electronic check",
            "Bank transfer (automatic)", "Credit card (automatic)",
        ],
        "MonthlyCharges": [29.85, 56.95, 53.85, 20.05, 89.10],
        "TotalCharges": [29.85, 1889.50, 108.15, 900.25, 5340.00],
        "Churn": [0, 0, 1, 0, 1],
    })


class TestCreateTenureBins:
    """Tests for create_tenure_bins function."""

    def test_creates_tenure_bin_column(self, sample_cleaned_df):
        result = create_tenure_bins(sample_cleaned_df)
        assert "tenure_bin" in result.columns

    def test_correct_bin_assignments(self, sample_cleaned_df):
        result = create_tenure_bins(sample_cleaned_df)
        assert result.loc[0, "tenure_bin"] == "0-12m"    # tenure=1
        assert result.loc[1, "tenure_bin"] == "24-48m"    # tenure=34
        assert result.loc[4, "tenure_bin"] == "48-72m"    # tenure=60

    def test_does_not_modify_original(self, sample_cleaned_df):
        original_cols = set(sample_cleaned_df.columns)
        create_tenure_bins(sample_cleaned_df)
        assert set(sample_cleaned_df.columns) == original_cols


class TestCreateChargeFeatures:
    """Tests for create_charge_features function."""

    def test_creates_avg_monthly_charge(self, sample_cleaned_df):
        result = create_charge_features(sample_cleaned_df)
        assert "AvgMonthlyCharge" in result.columns

    def test_creates_charge_ratio(self, sample_cleaned_df):
        result = create_charge_features(sample_cleaned_df)
        assert "ChargeRatio" in result.columns

    def test_avg_monthly_charge_calculation(self, sample_cleaned_df):
        result = create_charge_features(sample_cleaned_df)
        # Row 1: TotalCharges=1889.50, tenure=34 -> 1889.50/34 ≈ 55.573
        expected = 1889.50 / 34
        assert abs(result.loc[1, "AvgMonthlyCharge"] - expected) < 0.01

    def test_zero_tenure_uses_monthly(self):
        df = pd.DataFrame({
            "tenure": [0],
            "MonthlyCharges": [50.0],
            "TotalCharges": [0.0],
        })
        result = create_charge_features(df)
        assert result.loc[0, "AvgMonthlyCharge"] == 50.0

    def test_charge_ratio_range(self, sample_cleaned_df):
        result = create_charge_features(sample_cleaned_df)
        assert (result["ChargeRatio"] >= 0).all()


class TestCreateServiceCount:
    """Tests for create_service_count function."""

    def test_creates_num_services(self, sample_cleaned_df):
        result = create_service_count(sample_cleaned_df)
        assert "NumServices" in result.columns

    def test_creates_charge_per_service(self, sample_cleaned_df):
        result = create_service_count(sample_cleaned_df)
        assert "ChargePerService" in result.columns

    def test_service_count_non_negative(self, sample_cleaned_df):
        result = create_service_count(sample_cleaned_df)
        assert (result["NumServices"] >= 0).all()

    def test_charge_per_service_non_negative(self, sample_cleaned_df):
        result = create_service_count(sample_cleaned_df)
        assert (result["ChargePerService"] >= 0).all()


class TestEncodeCategoricalFeatures:
    """Tests for encode_categorical_features function."""

    def test_no_object_columns_remain(self, sample_cleaned_df):
        result = encode_categorical_features(sample_cleaned_df)
        assert len(result.select_dtypes(include=["object"]).columns) == 0

    def test_increases_column_count(self, sample_cleaned_df):
        original_cols = len(sample_cleaned_df.columns)
        result = encode_categorical_features(sample_cleaned_df)
        assert len(result.columns) >= original_cols  # One-hot adds columns

    def test_preserves_row_count(self, sample_cleaned_df):
        result = encode_categorical_features(sample_cleaned_df)
        assert len(result) == len(sample_cleaned_df)


class TestScaleNumericFeatures:
    """Tests for scale_numeric_features function."""

    def test_returns_dataframe_and_scaler(self, sample_cleaned_df):
        # Only keep numeric columns for this test
        numeric_df = sample_cleaned_df.select_dtypes(include=[np.number])
        df_scaled, scaler = scale_numeric_features(numeric_df, exclude=["Churn"])
        assert isinstance(df_scaled, pd.DataFrame)
        assert scaler is not None

    def test_excludes_target_from_scaling(self, sample_cleaned_df):
        numeric_df = sample_cleaned_df.select_dtypes(include=[np.number])
        df_scaled, _ = scale_numeric_features(numeric_df, exclude=["Churn"])
        # Churn values should remain 0 and 1
        assert set(df_scaled["Churn"].unique()) <= {0, 1}

    def test_scaled_features_approximately_standardized(self):
        df = pd.DataFrame({
            "A": [10, 20, 30, 40, 50],
            "target": [0, 1, 0, 1, 0],
        })
        df_scaled, _ = scale_numeric_features(df, exclude=["target"])
        assert abs(df_scaled["A"].mean()) < 1e-10
        assert abs(df_scaled["A"].std(ddof=0) - 1.0) < 0.01
