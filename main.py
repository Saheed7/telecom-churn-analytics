"""CLI entrypoint for the Telecom Customer Churn Analytics Pipeline.

Usage:
    python main.py                          # Run full pipeline
    python main.py --skip-eda               # Skip EDA plots
    python main.py --data path/to/data.csv  # Custom data path
"""

import argparse
import logging
import sys
from pathlib import Path

from src.data_ingestion import load_data, generate_quality_report
from src.data_cleaning import clean_data
from src.eda import run_eda
from src.feature_engineering import engineer_features
from src.model_training import (
    split_data, apply_smote, train_models, select_best_model, save_model,
)
from src.model_evaluation import generate_evaluation_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Telecom Customer Churn Analytics Pipeline"
    )
    parser.add_argument(
        "--data", type=str, default="data/raw/telco_churn.csv",
        help="Path to the raw CSV data file (default: data/raw/telco_churn.csv)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="reports/figures",
        help="Directory for saving figures and reports",
    )
    parser.add_argument(
        "--model-dir", type=str, default="models",
        help="Directory for saving trained models",
    )
    parser.add_argument(
        "--skip-eda", action="store_true",
        help="Skip EDA visualization step",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE oversampling",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)

    logger.info("=" * 60)
    logger.info("TELECOM CUSTOMER CHURN ANALYTICS PIPELINE")
    logger.info("=" * 60)

    # Step 1: Data Ingestion
    logger.info("\n[STEP 1/6] Data Ingestion")
    df_raw = load_data(args.data)
    quality_report = generate_quality_report(df_raw)
    print(quality_report)

    # Step 2: Data Cleaning
    logger.info("\n[STEP 2/6] Data Cleaning")
    df_clean = clean_data(df_raw)

    # Step 3: Exploratory Data Analysis
    if not args.skip_eda:
        logger.info("\n[STEP 3/6] Exploratory Data Analysis")
        chi_results = run_eda(df_clean, output_dir)
        print("\nChi-Square Test Results:")
        print(chi_results.to_string(index=False))
    else:
        logger.info("\n[STEP 3/6] Skipping EDA (--skip-eda)")

    # Step 4: Feature Engineering
    logger.info("\n[STEP 4/6] Feature Engineering")
    df_features, scaler = engineer_features(df_clean, scale=True)

    # Step 5: Model Training
    logger.info("\n[STEP 5/6] Model Training")
    X_train, X_test, y_train, y_test = split_data(df_features)

    if not args.no_smote:
        X_train, y_train = apply_smote(X_train, y_train)

    trained_models = train_models(X_train, y_train)
    best_name, best_model = select_best_model(trained_models)

    model_path = model_dir / "best_model.joblib"
    save_model(best_model, model_path)

    # Step 6: Model Evaluation
    logger.info("\n[STEP 6/6] Model Evaluation")
    results = generate_evaluation_report(trained_models, X_test, y_test, output_dir)

    print("\n" + "=" * 60)
    print("MODEL COMPARISON RESULTS")
    print("=" * 60)
    print(results.to_string())
    print(f"\nBest Model: {best_name}")
    print(f"Model saved to: {model_path}")
    print(f"Figures saved to: {output_dir}/")

    logger.info("Pipeline complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
