
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# Load data and trained pipeline
df = pd.read_csv("credit_risk_dataset.csv")
model = joblib.load("loan_model.joblib")

target = "loan_status"
X = df.drop(columns=[target]).copy()
y = df[target].astype(int)

print("\n=== GROUP DEFAULT RATES ===")
print("1 = default, 0 = no default")

# Compare default rates for available groups
for col in [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
]:
    if col in df.columns:
        print(f"\n{col}")
        print(
            df.groupby(col, observed=True)[target]
            .agg(applicants="count", default_rate="mean")
            .assign(
                default_rate_pct=lambda t: (t["default_rate"] * 100).round(2)
            )
        )

# SHAP on a small sample for speed
sample = X.sample(min(150, len(X)), random_state=42)

preprocessor = model.named_steps["preprocessor"]
classifier = model.named_steps["classifier"]

transformed = preprocessor.transform(sample)
if hasattr(transformed, "toarray"):
    transformed = transformed.toarray()

feature_names = preprocessor.get_feature_names_out()

print("\nCalculating SHAP values...")
explainer = shap.TreeExplainer(classifier)
values = explainer.shap_values(transformed)

# Handle common SHAP output formats
if isinstance(values, list):
    values = values[1]
elif getattr(values, "ndim", 0) == 3:
    values = values[:, :, 1]

importance = pd.DataFrame({
    "feature": feature_names,
    "mean_abs_shap": abs(values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False)

print("\n=== TOP SHAP FEATURES ===")
print(importance.head(10).to_string(index=False))
importance.to_csv("shap_importance.csv", index=False)

# Save a plot
plt.figure()
importance.head(10).sort_values("mean_abs_shap").plot(
    x="feature",
    y="mean_abs_shap",
    kind="barh",
    legend=False,
)
plt.xlabel("Mean absolute SHAP value")
plt.ylabel("Feature")
plt.title("Global Model Explainability")
plt.tight_layout()
plt.savefig("shap_summary.png", dpi=150)
plt.close()

print("\nSaved shap_importance.csv and shap_summary.png")
