
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Meridian | Explainable Loan AI",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 Explainable Loan Risk Dashboard")
st.caption("Deviathon 2.0 | DEV-P02 | Meridian")

st.warning(
    "Educational prototype. This model predicts default risk, "
    "not whether a bank should approve a loan."
)

# Load the trained model
try:
    model = joblib.load("loan_model.joblib")
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

# Confirm this is the newer pipeline model
if not hasattr(model, "named_steps"):
    st.error(
        "The saved model is from the older training script. "
        "Run python train_model.py to retrain it."
    )
    st.stop()

st.header("📊 Model Overview")

col1, col2, col3 = st.columns(3)
col1.metric("Model", "Random Forest")
col2.metric("Target", "Loan default")
col3.metric("Explainability", "SHAP")

st.divider()

# Show SHAP chart
st.header("🔍 Explainable AI")

chart_path = Path("shap_summary.png")

if chart_path.exists():
    st.image(
        str(chart_path),
        caption="Global SHAP feature importance",
        width='stretch',
    )
else:
    st.info(
        "SHAP chart not found. Run "
        "python analyze_model.py to generate it."
    )

# Show SHAP feature importance table
importance_path = Path("shap_importance.csv")

if importance_path.exists():
    importance = pd.read_csv(importance_path)

    st.subheader("Most influential model features")
    st.dataframe(
        importance.head(10),
        width='stretch',
        hide_index=True,
    )
else:
    st.info("Run analyze_model.py to generate the feature report.")

st.divider()

# Group-level default rate analysis
st.header("⚖️ Group Default-Rate Comparison")

try:
    df = pd.read_csv("credit_risk_dataset.csv")

    target = "loan_status"

    if target not in df.columns:
        st.error("The dataset does not contain loan_status.")
    else:
        group_options = [
            "person_home_ownership",
            "loan_intent",
            "loan_grade",
        ]

        available = [c for c in group_options if c in df.columns]

        if available:
            selected = st.selectbox(
                "Compare groups by",
                available,
            )

            summary = (
                df.groupby(selected, observed=True)[target]
                .agg(["count", "mean"])
                .reset_index()
            )

            summary.columns = [
                "Group", "Applicants", "Default rate"
            ]
            summary["Default rate (%)"] = (
                summary["Default rate"] * 100
            ).round(2)

            st.dataframe(
                summary[
                    ["Group", "Applicants", "Default rate (%)"]
                ],
                width='stretch',
                hide_index=True,
            )

            st.bar_chart(
                summary.set_index("Group")["Default rate (%)"]
            )

            st.caption(
                "These are observed default-rate differences, "
                "not proof of discrimination. The dataset does "
                "not include gender or region."
            )

except FileNotFoundError:
    st.error("credit_risk_dataset.csv was not found.")

st.divider()

st.caption(
    "Research prototype only. Predictions and group comparisons "
    "must be validated before any real lending use."
)

