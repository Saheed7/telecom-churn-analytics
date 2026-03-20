"""Tests for the model training module."""

import pytest
import pandas as pd
import numpy as np

from src.model_training import split_data, apply_smote, select_best_model


@pytest.fixture
def sample_feature_df():
    """Create a sample feature-engineered dataframe for training tests."""
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "feature_1": np.random.randn(n),
        "feature_2": np.random.randn(n),
        "feature_3": np.random.randn(n),
        "feature_4": np.random.randn(n),
        "feature_5": np.random.randn(n),
        "Churn": np.random.choice([0, 1], size=n, p=[0.73, 0.27]),
    })


class TestSplitData:
    """Tests for split_data function."""

    def test_returns_four_elements(self, sample_feature_df):
        result = split_data(sample_feature_df)
        assert len(result) == 4

    def test_correct_split_ratio(self, sample_feature_df):
        X_train, X_test, y_train, y_test = split_data(sample_feature_df, test_size=0.2)
        total = len(X_train) + len(X_test)
        assert total == len(sample_feature_df)
        assert abs(len(X_test) / total - 0.2) < 0.05

    def test_stratified_split(self, sample_feature_df):
        _, _, y_train, y_test = split_data(sample_feature_df)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        # Stratification should keep rates similar
        assert abs(train_rate - test_rate) < 0.05

    def test_target_not_in_features(self, sample_feature_df):
        X_train, X_test, _, _ = split_data(sample_feature_df)
        assert "Churn" not in X_train.columns
        assert "Churn" not in X_test.columns

    def test_reproducible_with_same_seed(self, sample_feature_df):
        X1, _, _, _ = split_data(sample_feature_df, random_state=42)
        X2, _, _, _ = split_data(sample_feature_df, random_state=42)
        pd.testing.assert_frame_equal(X1, X2)


class TestApplySmote:
    """Tests for apply_smote function."""

    def test_balances_classes(self, sample_feature_df):
        X_train, _, y_train, _ = split_data(sample_feature_df)
        X_resampled, y_resampled = apply_smote(X_train, y_train)

        class_counts = pd.Series(y_resampled).value_counts()
        assert class_counts[0] == class_counts[1]

    def test_increases_sample_count(self, sample_feature_df):
        X_train, _, y_train, _ = split_data(sample_feature_df)
        X_resampled, y_resampled = apply_smote(X_train, y_train)
        assert len(X_resampled) >= len(X_train)

    def test_preserves_feature_names(self, sample_feature_df):
        X_train, _, y_train, _ = split_data(sample_feature_df)
        X_resampled, _ = apply_smote(X_train, y_train)
        assert list(X_resampled.columns) == list(X_train.columns)


class TestSelectBestModel:
    """Tests for select_best_model function."""

    def test_selects_highest_scoring_model(self):
        """Test with mock GridSearchCV objects."""

        class MockGrid:
            def __init__(self, score, estimator):
                self.best_score_ = score
                self.best_estimator_ = estimator

        models = {
            "ModelA": MockGrid(0.75, "estimator_a"),
            "ModelB": MockGrid(0.88, "estimator_b"),
            "ModelC": MockGrid(0.82, "estimator_c"),
        }

        name, model = select_best_model(models)
        assert name == "ModelB"
        assert model == "estimator_b"
