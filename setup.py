"""Package setup for Telecom Customer Churn Analytics Pipeline."""

from setuptools import setup, find_packages

setup(
    name="telecom-churn-analytics",
    version="1.0.0",
    description="End-to-end data science pipeline for telecom customer churn prediction",
    author="Yakub Kayode Saheed",
    author_email="kayodesaheed@gmail.com",
    url="https://github.com/Saheed7/telecom-churn-analytics",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "scikit-learn>=1.3.0",
        "xgboost>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "shap>=0.42.0",
        "imbalanced-learn>=0.11.0",
        "streamlit>=1.28.0",
        "joblib>=1.3.0",
    ],
    extras_require={
        "dev": ["pytest>=7.4.0", "pytest-cov>=4.1.0", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
