
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
)

# Load the larger credit-risk dataset
df = pd.read_csv("credit_risk_dataset.csv")
df.columns = df.columns.str.strip()

# Target: 1 = loan default, 0 = no default
target = "loan_status"

if target not in df.columns:
    raise ValueError(f"Missing target column: {target}")

df = df.dropna(subset=[target]).copy()
df[target] = df[target].astype(int)

X = df.drop(columns=[target])
y = df[target]

# Identify feature types
numeric_cols = X.select_dtypes(include="number").columns.tolist()
categorical_cols = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

# Handle missing values and encode categories
preprocessor = ColumnTransformer([
    (
        "numeric",
        SimpleImputer(strategy="median"),
        numeric_cols,
    ),
    (
        "categorical",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]),
        categorical_cols,
    ),
])

# Split before training to evaluate on unseen rows
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=150,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )),
])

model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)[:, 1]

print("\n--- CREDIT RISK MODEL EVALUATION ---")
print("Dataset rows:", len(df))
print("Training rows:", len(X_train))
print("Testing rows:", len(X_test))
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("ROC-AUC:", round(roc_auc_score(y_test, probabilities), 4))
print(classification_report(y_test, predictions, zero_division=0))

# Save the complete preprocessing + model pipeline
joblib.dump(model, "loan_model.joblib")
print("\nSaved model to loan_model.joblib")
print("Target meaning: 1 = default; 0 = no default.")
