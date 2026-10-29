
import streamlit as st
import pandas as pd
import joblib
import pytesseract
from PIL import Image
from pathlib import Path
from io import BytesIO

st.set_page_config(
    page_title="Fairlens | Loan Risk Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown("""
<style>
.stApp { background: #F6F5EF; color: #173D32; }
section[data-testid="stSidebar"] {
    background: #173D32;
}
section[data-testid="stSidebar"] * { color: #FFFFFF !important; }
.hero {
    background: #173D32; color: #FFFFFF; padding: 28px;
    border-radius: 18px; margin-bottom: 22px;
}
.hero h1 { color: #FFFFFF; margin: 0; }
.hero p { color: #DCE8DE; margin-bottom: 0; }
div[data-testid="stMetric"] {
    background: #FFFFFF; padding: 16px; border-radius: 12px;
    border: 1px solid #E1E5DC;
}
.stButton button {
    background: #245D49; color: white; border-radius: 9px;
    border: 0; padding: 9px 18px;
}
</style>
""", unsafe_allow_html=True)

# ---------- Load model ----------
@st.cache_resource
def load_model():
    return joblib.load("loan_model.joblib")

try:
    model = load_model()
except Exception as exc:
    st.error(f"Could not load the trained model: {exc}")
    st.stop()

if not hasattr(model, "named_steps"):
    st.error("The model must be a trained preprocessing pipeline. Run train_model.py.")
    st.stop()

st.sidebar.markdown("## 🏦 FAIRLENS")
st.sidebar.caption("Explainable Loan Risk Intelligence")
page = st.sidebar.radio(
    "NAVIGATION",
    ["Applicant Assessment", "Model Report", "Fairness Audit"],
)
st.sidebar.divider()
st.sidebar.caption("DEV-P02 · Deviathon 2.0")
st.sidebar.caption("Educational prototype — not a lending decision.")

st.markdown("""
<div class="hero">
<h1>Fairlens</h1>
<p>Understand credit risk. Explain every estimate. Assess responsibly.</p>
</div>
""", unsafe_allow_html=True)

st.warning(
    "This model predicts the likelihood of historical loan default patterns. "
    "It does not determine loan approval and is not validated for real lending."
)

# ---------- Applicant assessment ----------
if page == "Applicant Assessment":
    st.subheader("Applicant profile")
    st.write(
        "Enter financial information below. Fields marked with * are required. "
        "PAN OCR is optional and never supplies financial or credit data."
    )

    with st.form("assessment_form"):
        st.markdown("#### Personal information")
        a, b, c = st.columns(3)
        applicant_name = a.text_input("Applicant name")
        age = b.number_input("Age *", min_value=18, max_value=100, value=25)
        employment = c.number_input(
            "Employment length (years) *",
            min_value=0.0, max_value=60.0, value=2.0, step=0.5
        )

        st.markdown("#### Income and requested loan")
        a, b, c = st.columns(3)
        monthly_income = a.number_input(
            "Monthly income (₹) *", min_value=1, value=50000, step=5000
        )
        co_income = b.number_input(
            "Co-applicant monthly income (₹)",
            min_value=0, value=0, step=5000
        )
        loan_amount = c.number_input(
            "Loan amount requested (₹) *",
            min_value=1000, value=200000, step=10000
        )

        a, b, c = st.columns(3)
        loan_term = a.selectbox("Loan term", [12, 24, 36, 48, 60, 84, 120, 180, 240])
        home = b.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
        intent = c.selectbox(
            "Loan purpose",
            ["EDUCATION", "PERSONAL", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )

        st.markdown("#### Credit profile")
        a, b, c = st.columns(3)
        grade = a.selectbox("Loan grade", ["A", "B", "C", "D", "E", "F", "G"])
        interest_rate = b.number_input(
            "Interest rate (%) *", min_value=0.1, max_value=50.0,
            value=12.0, step=0.5
        )
        credit_history = c.number_input(
            "Credit history length (years)",
            min_value=0, max_value=60, value=3
        )

        a, b = st.columns(2)
        previous_default = a.selectbox(
            "Previous default on file?", ["N", "Y"]
        )
        existing_obligations = b.number_input(
            "Existing monthly debt payments (₹)",
            min_value=0, value=0, step=1000
        )

        submitted = st.form_submit_button(
            "Assess default risk", type="primary", use_container_width=True
        )

    st.divider()
    st.subheader("Optional PAN card OCR")
    st.caption(
        "Upload only if needed. OCR runs locally on this computer. "
        "The app does not save the uploaded image or extracted PAN number. "
        "Extracted text is unverified and may be incorrect."
    )
    pan_file = st.file_uploader(
        "Upload PAN image (PNG/JPG)",
        type=["png", "jpg", "jpeg"],
        key="pan_upload",
    )

    if pan_file is not None:
        try:
            pytesseract.pytesseract.tesseract_cmd = (
                r"C:\Program Files\Tesseract-OCR\tesseract.exe"
            )
            image = Image.open(BytesIO(pan_file.getvalue())).convert("RGB")
            with st.spinner("Reading image locally..."):
                text = pytesseract.image_to_string(image)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            # Do not show or retain the PAN number.
            safe_lines = [
                line for line in lines
                if not __import__("re").search(
                    r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", line.upper()
                )
            ]
            st.info(
                "Check the extracted text below and verify it manually. "
                "PAN number patterns are hidden."
            )
            st.text_area(
                "Extracted text (unverified)",
                "\n".join(safe_lines[:12]),
                height=140,
            )
        except Exception as exc:
            st.error(f"OCR could not read this image: {exc}")

    if submitted:
        if monthly_income <= 0:
            st.error("Monthly income must be greater than zero.")
            st.stop()

        annual_income = monthly_income * 12
        total_monthly_income = monthly_income + co_income
        monthly_loan_payment = (
            loan_amount * (interest_rate / 1200) /
            (1 - (1 + interest_rate / 1200) ** (-loan_term))
            if interest_rate > 0 else loan_amount / loan_term
        )
        debt_to_income = (
            (existing_obligations + monthly_loan_payment)
            / max(total_monthly_income, 1)
        )
        loan_percent_income = loan_amount / max(annual_income, 1)

        # Exact feature names required by the existing trained model.
        applicant = pd.DataFrame([{
            "person_age": int(age),
            "person_income": float(annual_income),
            "person_home_ownership": home,
            "person_emp_length": float(employment),
            "loan_intent": intent,
            "loan_grade": grade,
            "loan_amnt": float(loan_amount),
            "loan_int_rate": float(interest_rate),
            "loan_percent_income": float(loan_percent_income),
            "cb_person_default_on_file": previous_default,
            "cb_person_cred_hist_length": int(credit_history),
        }])

        try:
            risk_probability = float(model.predict_proba(applicant)[0][1])
            if risk_probability < 0.20:
                risk_label = "Lower estimated risk"
            elif risk_probability < 0.50:
                risk_label = "Moderate estimated risk"
            else:
                risk_label = "Higher estimated risk"

            st.markdown("### Assessment result")
            m1, m2, m3 = st.columns(3)
            m1.metric("Estimated default risk", f"{risk_probability:.1%}")
            m2.metric("Risk category", risk_label)
            m3.metric("Monthly payment estimate", f"₹{monthly_loan_payment:,.0f}")

            st.progress(risk_probability)
            st.caption(
                "Risk bands are illustrative, not bank-approved thresholds. "
                "The model's probability may be poorly calibrated for new applicants."
            )

            st.markdown("#### Financial overview")
            x, y, z = st.columns(3)
            x.metric("Annual applicant income", f"₹{annual_income:,.0f}")
            y.metric("Combined monthly income", f"₹{total_monthly_income:,.0f}")
            z.metric("Estimated payment-to-income", f"{debt_to_income:.1%}")

            st.markdown("#### What the model considered")
            st.dataframe(
                pd.DataFrame({
                    "Input": applicant.columns,
                    "Value used": [str(v) for v in applicant.iloc[0].tolist()],
                }),
                use_container_width=True,
                hide_index=True,
            )

            st.info(
                "The estimate uses age, annual applicant income, home ownership, "
                "employment length, loan purpose, grade, amount, interest rate, "
                "loan-to-income ratio, previous-default indicator and credit-history "
                "length. Co-applicant income, loan term and existing obligations "
                "are used only for the displayed affordability calculations, not "
                "by the trained risk model."
            )

            st.markdown("#### Explanation and next steps")
            if risk_probability >= 0.50:
                st.write(
                    "The model estimates relatively high default risk for this input. "
                    "Review the input values and the factors shown above. This is "
                    "not a rejection recommendation."
                )
            elif risk_probability >= 0.20:
                st.write(
                    "The model estimates moderate default risk. Verify the financial "
                    "details and consider additional, independently validated evidence."
                )
            else:
                st.write(
                    "The model estimates lower default risk relative to its learned "
                    "patterns. This is not a guarantee of repayment or loan eligibility."
                )

            st.caption(
                f"Applicant label: {applicant_name or 'Not provided'}. "
                "No applicant record is saved by this app."
            )
        except Exception as exc:
            st.error(f"Prediction failed. Check the inputs and model: {exc}")

# ---------- Model report ----------
elif page == "Model Report":
    st.subheader("Model performance and explainability")
    st.write(
        "These metrics describe evaluation results from the existing project. "
        "They are not automatically recomputed for this screen."
    )

    metrics = {
        "Accuracy": "91.41%",
        "Precision (default class)": "83.26%",
        "Recall (default class)": "75.88%",
        "F1 score (default class)": "79.40%",
        "ROC-AUC": "93.14%",
    }
    cols = st.columns(len(metrics))
    for col, (label, value) in zip(cols, metrics.items()):
        col.metric(label, value)

    st.caption(
        "Use the held-out evaluation script to regenerate metrics after changing "
        "the dataset, features or model."
    )

    st.subheader("Global feature importance")
    chart_path = Path("shap_summary.png")
    importance_path = Path("shap_importance.csv")

    if chart_path.exists():
        st.image(str(chart_path), use_container_width=True)
    if importance_path.exists():
        st.dataframe(
            pd.read_csv(importance_path),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Limitations")
    st.write(
        "- The model predicts default, not loan approval.\n"
        "- Its historical training data may not represent current applicants.\n"
        "- Global SHAP importance does not explain an individual prediction.\n"
        "- The model is a prototype and must not be used for real lending decisions."
    )

# ---------- Fairness audit ----------
else:
    st.subheader("Fairness audit")
    st.write(
        "These group comparisons use a separate UCI credit-card-default dataset. "
        "They do not describe the main loan-risk model's demographic fairness."
    )

    for label, filename in [
        ("By sex", "fairness_by_sex.csv"),
        ("By age group", "fairness_by_age.csv"),
    ]:
        st.markdown(f"#### {label}")
        path = Path(filename)
        if path.exists():
            data = pd.read_csv(path)
            st.dataframe(data, use_container_width=True, hide_index=True)

            group_col = "SEX" if "SEX" in data.columns else (
                "age_group" if "age_group" in data.columns else None
            )
            rate_cols = [
                col for col in
                ["actual_default_rate", "predicted_default_rate"]
                if col in data.columns
            ]
            if group_col and rate_cols:
                chart = data.set_index(group_col)[rate_cols] * 100
                st.bar_chart(chart)
                st.caption("Rates are percentages.")
        else:
            st.warning(f"{filename} is missing. Run fairness_analysis.py first.")

    st.warning(
        "Group differences are not proof of discrimination. This audit is exploratory "
        "and does not establish legal or statistical fairness."
    )

st.divider()
st.caption(
    "FAIRLENS · Educational research prototype · No real lending decisions"
)
