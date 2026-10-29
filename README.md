# \# Fairlens — Explainable Loan Risk Dashboard

# 

# An educational machine learning project built for \*\*Deviathon 2.0 (DEV-P02)\*\*. Fairlens uses a Random Forest classifier to estimate loan-default risk and provides explainability insights through SHAP feature importance and applicant-level sensitivity analysis.

# 

# \## Features

# 

# \- Loan-default risk probability estimation

# \- Applicant-level input sensitivity analysis

# \- Global SHAP feature-importance visualization

# \- Group-level default-rate comparisons

# \- Model evaluation metrics and confusion matrix

# \- Interactive Streamlit dashboard

# 

# \## Tech Stack

# 

# Python · Streamlit · Pandas · NumPy · Scikit-learn · SHAP · Matplotlib

# 

# \## Project Setup

# 

# \### 1. Clone the repository

# 

# ```bash

# git clone https://github.com/Onewithgovinda/Fairlens.git

# cd Fairlens

# ```

# 

# \### 2. Create a virtual environment

# 

# ```bash

# python -m venv .venv

# ```

# 

# Activate it on Windows PowerShell:

# 

# ```powershell

# .\\.venv\\Scripts\\Activate.ps1

# ```

# 

# \### 3. Install dependencies

# 

# ```bash

# pip install -r requirements.txt

# ```

# 

# \### 4. Prepare the dataset

# 

# Obtain the credit-risk dataset from its original source and confirm its license permits your intended use. Place it in the project root as `credit\_risk\_dataset.csv`.

# 

# The dataset must contain a `loan\_status` column, where `1` represents default and `0` represents non-default.

# 

# \### 5. Train the model

# 

# ```bash

# python train\_model.py

# ```

# 

# This creates `loan\_model.joblib`.

# 

# \### 6. Generate explainability reports

# 

# ```bash

# python analyze\_model.py

# ```

# 

# \### 7. Run the dashboard

# 

# ```bash

# python -m streamlit run app.py

# ```

# 

# Open the local URL displayed in the terminal.

# 

# \## Model Evaluation

# 

# The current model evaluation uses an 80/20 stratified train/test split. In one run, the model achieved 91.41% accuracy and a ROC-AUC of 0.9314 on 6,517 held-out records. Results can vary if the dataset or training configuration changes.

# 

# \## Limitations

# 

# This is an educational prototype, not a loan-approval system. Model explanations are not proof of causation or fairness. The model requires further validation, calibration, bias analysis, and legal/compliance review before any real-world lending use.

#


