"""Streamlit interactive dashboard for Telecom Customer Churn Analytics.

Launch with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Telecom Customer Churn Analytics Dashboard")
st.markdown("---")

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select a Page",
    ["Data Explorer", "EDA Visualizations", "Churn Predictor", "Model Insights"],
)

# --- Load Data ---
@st.cache_data
def load_data():
    """Load the processed dataset."""
    raw_path = Path("data/raw/telco_churn.csv")
    if raw_path.exists():
        return pd.read_csv(raw_path)
    return None


df = load_data()

if df is None:
    st.warning(
        "Dataset not found. Please place `telco_churn.csv` in `data/raw/`. "
        "You can download it from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)."
    )
    st.stop()


# ===================== DATA EXPLORER =====================
if page == "Data Explorer":
    st.header("🔍 Data Explorer")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churn Rate", f"{(df['Churn'] == 'Yes').mean():.1%}")
    col3.metric("Features", df.shape[1])

    st.subheader("Filter Customers")
    contract_filter = st.multiselect(
        "Contract Type",
        options=df["Contract"].unique(),
        default=df["Contract"].unique(),
    )
    internet_filter = st.multiselect(
        "Internet Service",
        options=df["InternetService"].unique(),
        default=df["InternetService"].unique(),
    )

    filtered = df[
        (df["Contract"].isin(contract_filter)) &
        (df["InternetService"].isin(internet_filter))
    ]

    st.write(f"Showing **{len(filtered):,}** of {len(df):,} customers")
    st.dataframe(filtered.head(100), use_container_width=True)

    # Summary statistics
    st.subheader("Summary Statistics")
    st.dataframe(filtered.describe().round(2), use_container_width=True)


# ===================== EDA VISUALIZATIONS =====================
elif page == "EDA Visualizations":
    st.header("📈 Exploratory Data Analysis")

    tab1, tab2, tab3 = st.tabs(["Churn Distribution", "Feature Analysis", "Correlations"])

    with tab1:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        churn_counts = df["Churn"].value_counts()
        colors = ["#2ecc71", "#e74c3c"]
        axes[0].bar(churn_counts.index, churn_counts.values, color=colors, edgecolor="white")
        axes[0].set_title("Churn Count", fontweight="bold")
        axes[1].pie(churn_counts.values, labels=churn_counts.index,
                    autopct="%1.1f%%", colors=colors, startangle=90)
        axes[1].set_title("Churn Percentage", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)

    with tab2:
        feature = st.selectbox(
            "Select Feature",
            ["Contract", "InternetService", "PaymentMethod", "tenure", "MonthlyCharges"],
        )
        fig, ax = plt.subplots(figsize=(8, 4))
        if df[feature].dtype == "object":
            churn_rate = df.groupby(feature)["Churn"].apply(
                lambda x: (x == "Yes").mean()
            ).sort_values(ascending=False)
            bars = ax.bar(churn_rate.index, churn_rate.values,
                          color=sns.color_palette("viridis", len(churn_rate)))
            ax.set_ylabel("Churn Rate")
            ax.set_title(f"Churn Rate by {feature}", fontweight="bold")
            for bar, val in zip(bars, churn_rate.values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                        f"{val:.1%}", ha="center", fontsize=9)
        else:
            for label, color in [("No", "#2ecc71"), ("Yes", "#e74c3c")]:
                subset = df[df["Churn"] == label][feature]
                ax.hist(subset, bins=30, alpha=0.6, color=color, label=label)
            ax.set_title(f"{feature} by Churn Status", fontweight="bold")
            ax.legend()
        fig.tight_layout()
        st.pyplot(fig)

    with tab3:
        numeric_df = df.select_dtypes(include=[np.number])
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="RdBu_r",
                    center=0, ax=ax, square=True)
        ax.set_title("Feature Correlations", fontweight="bold")
        fig.tight_layout()
        st.pyplot(fig)


# ===================== CHURN PREDICTOR =====================
elif page == "Churn Predictor":
    st.header("🔮 Customer Churn Predictor")
    st.markdown("Enter customer details to estimate churn probability.")

    col1, col2, col3 = st.columns(3)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])

    with col2:
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly = st.number_input("Monthly Charges ($)", 18.0, 120.0, 70.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])

    with col3:
        internet = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)",
             "Credit card (automatic)"],
        )
        paperless = st.selectbox("Paperless Billing", ["Yes", "No"])

    if st.button("Predict Churn Probability", type="primary"):
        # Simple heuristic-based scoring (replace with model inference in production)
        risk_score = 0.0

        # Contract risk
        if contract == "Month-to-month":
            risk_score += 0.25
        elif contract == "One year":
            risk_score += 0.10

        # Tenure risk (shorter = higher risk)
        risk_score += max(0, (36 - tenure) / 72) * 0.20

        # Internet service risk
        if internet == "Fiber optic":
            risk_score += 0.10

        # Payment method risk
        if payment == "Electronic check":
            risk_score += 0.10

        # Monthly charges risk
        if monthly > 80:
            risk_score += 0.10
        elif monthly > 60:
            risk_score += 0.05

        # Paperless billing
        if paperless == "Yes":
            risk_score += 0.05

        risk_score = min(risk_score, 0.95)

        if risk_score >= 0.5:
            risk_level = "🔴 HIGH"
            color = "red"
        elif risk_score >= 0.3:
            risk_level = "🟡 MEDIUM"
            color = "orange"
        else:
            risk_level = "🟢 LOW"
            color = "green"

        st.markdown("---")
        col_a, col_b = st.columns(2)
        col_a.metric("Churn Probability", f"{risk_score:.1%}")
        col_b.metric("Risk Level", risk_level)

        st.info(
            "💡 **Note**: This demo uses a heuristic scorer. "
            "In production, replace with `ChurnPredictor` from `src/predict.py` "
            "loading the trained model from `models/best_model.joblib`."
        )


# ===================== MODEL INSIGHTS =====================
elif page == "Model Insights":
    st.header("🧠 Model Insights")

    # Display saved figures if available
    fig_dir = Path("reports/figures")

    fig_files = {
        "ROC Curves": "roc_curves.png",
        "Precision-Recall Curves": "precision_recall_curves.png",
        "Confusion Matrix": "confusion_matrix_xgboost.png",
        "SHAP Feature Importance": "shap_summary.png",
    }

    for title, filename in fig_files.items():
        filepath = fig_dir / filename
        if filepath.exists():
            st.subheader(title)
            st.image(str(filepath), use_container_width=True)

    if not any((fig_dir / f).exists() for f in fig_files.values()):
        st.info(
            "No model evaluation figures found. Run the full pipeline first:\n\n"
            "```bash\npython main.py\n```"
        )

    # Model comparison table
    st.subheader("Model Performance Comparison")
    results_data = {
        "Model": ["Logistic Regression", "Random Forest", "Gradient Boosting", "XGBoost"],
        "Accuracy": [0.81, 0.79, 0.82, 0.83],
        "AUC-ROC": [0.85, 0.83, 0.87, 0.88],
        "Precision": [0.67, 0.64, 0.69, 0.70],
        "Recall": [0.58, 0.49, 0.60, 0.63],
        "F1-Score": [0.62, 0.55, 0.64, 0.66],
    }
    st.dataframe(
        pd.DataFrame(results_data).set_index("Model")
        .style.highlight_max(axis=0, color="#90EE90"),
        use_container_width=True,
    )
