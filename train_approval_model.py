import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("loan_data.csv")
df.columns = df.columns.str.strip()

target = "approved"
required = [
    "income",
    "credit_score",
    "loan_amount",
    "employment_years",
    "existing_debt",
    target,
]

if not set(required).issubset(df.columns):
    raise ValueError("loan_data.csv is missing required columns.")

df = df[required].dropna().copy()
X = df.drop(columns=[target])
y = df[target].astype(int)

if set(y.unique()) != {0, 1}:
    raise ValueError("The approved column must contain both 0 and 1.")

model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )),
])

model.fit(X, y)
joblib.dump(model, "approval_model.joblib")

print("Approval model saved to approval_model.joblib")
print("Training rows:", len(df))
print("Target: 1 = approved, 0 = rejected")
print("WARNING: This 20-row dataset is for prototype testing only.")
