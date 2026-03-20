# Telecom Customer Churn Analytics Pipeline

[![CI Tests](https://github.com/Saheed7/telecom-churn-analytics/actions/workflows/ci.yml/badge.svg)](https://github.com/Saheed7/telecom-churn-analytics/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end data science pipeline for predicting telecom customer churn from raw data ingestion and exploratory data analysis (EDA) through feature engineering, model training, evaluation, and interactive dashboard deployment.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Pipeline Stages](#pipeline-stages)
- [Results](#results)
- [Dashboard](#dashboard)
- [Testing](#testing)
- [Technologies](#technologies)
- [License](#license)

---

## Overview

Customer churn is one of the most critical challenges facing telecom companies, with the cost of acquiring a new customer being **5–7x higher** than retaining an existing one. This project builds a production-ready analytics pipeline that:

1. **Ingests and cleans** raw telecom customer data (7,043 records, 21 features)
2. **Explores** the data through comprehensive EDA with statistical tests and rich visualizations
3. **Engineers** predictive features (tenure bins, monthly-to-total charge ratio, service bundles, etc.)
4. **Trains** multiple ML models (Logistic Regression, Random Forest, Gradient Boosting, XGBoost)
5. **Evaluates** models with cross-validation, ROC-AUC, precision-recall curves, and SHAP interpretability
6. **Deploys** a Streamlit dashboard for interactive exploration and real-time prediction

---

## Project Structure

```
telecom-churn-analytics/
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI pipeline
├── data/
│   ├── raw/                        # Original dataset
│   └── processed/                  # Cleaned/engineered features
├── models/                         # Serialized trained models
├── notebooks/
│   └── 01_eda_and_modeling.ipynb   # Full EDA + modeling notebook
├── reports/
│   └── figures/                    # Generated plots and charts
├── src/
│   ├── __init__.py
│   ├── data_ingestion.py           # Data loading and validation
│   ├── data_cleaning.py            # Missing values, type casting, outliers
│   ├── eda.py                      # Exploratory data analysis & visualization
│   ├── feature_engineering.py      # Feature creation and transformation
│   ├── model_training.py           # Train and compare ML models
│   ├── model_evaluation.py         # Evaluation metrics, SHAP, reports
│   └── predict.py                  # Inference on new data
├── tests/
│   ├── __init__.py
│   ├── test_data_cleaning.py
│   ├── test_feature_engineering.py
│   └── test_model_training.py
├── app.py                          # Streamlit interactive dashboard
├── main.py                         # CLI entrypoint for full pipeline
├── requirements.txt
├── setup.py
├── LICENSE
└── README.md
```

---

## Key Features

- **Comprehensive EDA**: Univariate/bivariate analysis, correlation heatmaps, chi-square tests, distribution plots, churn segmentation
- **Robust Data Wrangling**: Handles missing values, type inconsistencies, outlier detection, and class imbalance (SMOTE)
- **Feature Engineering**: Tenure bucketing, service bundle aggregation, charge ratios, interaction features
- **Multiple ML Models**: Logistic Regression, Random Forest, Gradient Boosting, XGBoost — with hyperparameter tuning via GridSearchCV
- **Model Interpretability**: SHAP summary and dependence plots for transparent, explainable predictions
- **Interactive Dashboard**: Streamlit app for real-time churn probability estimation and data exploration
- **CI/CD**: Automated testing with pytest via GitHub Actions
- **Modular Design**: Clean, reusable Python modules following software engineering best practices

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Saheed7/telecom-churn-analytics.git
cd telecom-churn-analytics

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Run the full pipeline

```bash
python main.py
```

This executes all stages: data ingestion → cleaning → EDA → feature engineering → training → evaluation.

### Launch the interactive dashboard

```bash
streamlit run app.py
```

### Make predictions on new data

```python
from src.predict import ChurnPredictor

predictor = ChurnPredictor(model_path="models/best_model.joblib")
probability = predictor.predict_single({
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "tenure": 24,
    "MonthlyCharges": 79.85,
    "Contract": "One year",
    "InternetService": "Fiber optic",
    "PaymentMethod": "Electronic check",
    # ... other features
})
print(f"Churn Probability: {probability:.2%}")
```

---

## Pipeline Stages

### 1. Data Ingestion (`src/data_ingestion.py`)
- Loads CSV data with schema validation
- Performs initial data profiling (shape, dtypes, nulls, duplicates)
- Generates a data quality report

### 2. Data Cleaning (`src/data_cleaning.py`)
- Casts `TotalCharges` from string to float, handling whitespace entries
- Imputes missing values using median strategy
- Detects and handles outliers via IQR method
- Encodes binary categorical features

### 3. Exploratory Data Analysis (`src/eda.py`)
- Churn distribution analysis with count plots
- Correlation heatmap for numeric features
- Box plots and violin plots for charges by churn status
- Chi-square tests for categorical feature significance
- Tenure and contract type segmentation analysis
- Saves all figures to `reports/figures/`

### 4. Feature Engineering (`src/feature_engineering.py`)
- Creates tenure bins (0–12, 12–24, 24–48, 48–72 months)
- Computes `AvgMonthlyCharge = TotalCharges / tenure`
- Builds `NumServices` (count of active services per customer)
- Generates `ChargePerService = MonthlyCharges / NumServices`
- One-hot encodes multi-class categoricals with `pd.get_dummies()`
- Applies StandardScaler to numeric features

### 5. Model Training (`src/model_training.py`)
- Stratified train/test split (80/20)
- SMOTE oversampling for class imbalance
- Trains: Logistic Regression, Random Forest, Gradient Boosting, XGBoost
- Hyperparameter tuning with 5-fold cross-validated GridSearchCV
- Saves best model to `models/best_model.joblib`

### 6. Model Evaluation (`src/model_evaluation.py`)
- Classification report (precision, recall, F1-score)
- ROC-AUC curves for all models
- Precision-Recall curves
- Confusion matrix heatmaps
- SHAP summary plot for feature importance and interpretability
- Exports evaluation report to `reports/`

---

## Results

| Model | Accuracy | AUC-ROC | Precision | Recall | F1-Score |
|-------|----------|---------|-----------|--------|----------|
| Logistic Regression | 0.81 | 0.85 | 0.67 | 0.58 | 0.62 |
| Random Forest | 0.79 | 0.83 | 0.64 | 0.49 | 0.55 |
| Gradient Boosting | 0.82 | 0.87 | 0.69 | 0.60 | 0.64 |
| **XGBoost** | **0.83** | **0.88** | **0.70** | **0.63** | **0.66** |

**Top churn predictors** (via SHAP): Contract type, tenure, monthly charges, internet service type, payment method.

---

## Dashboard

The Streamlit dashboard provides:
- **Data Explorer**: Filter and view customer segments with interactive charts
- **EDA Visualizations**: Distribution plots, correlation analysis, churn breakdowns
- **Churn Predictor**: Input customer attributes and get real-time churn probability
- **Model Insights**: Feature importance rankings and SHAP explanations

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```

---

## Technologies

- **Python 3.10+** — Core language
- **Pandas, NumPy** — Data manipulation and numerical computing
- **Matplotlib, Seaborn** — Data visualization
- **Scikit-learn** — ML algorithms, preprocessing, evaluation
- **XGBoost** — Gradient boosting classifier
- **SHAP** — Model interpretability
- **Imbalanced-learn** — SMOTE for class imbalance
- **Streamlit** — Interactive dashboard
- **Joblib** — Model serialization
- **Pytest** — Testing framework
- **GitHub Actions** — CI/CD

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

## Author

**Yakub Kayode Saheed**
- [LinkedIn](https://www.linkedin.com/in/yakub-kayode-saheed-94468672/)
- [GitHub](https://github.com/Saheed7)
- [Google Scholar](https://scholar.google.com/citations?user=faYh6iIAAAAJ&hl=en)
