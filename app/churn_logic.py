"""
churn_logic.py
Core prediction and explanation logic for the churn app — kept separate from the
Streamlit UI code so it can be tested and reused independently.
"""

import numpy as np
import pandas as pd
import joblib
import shap
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pkl")
BACKGROUND_PATH = os.path.join(BASE_DIR, "background_sample.csv")

RAW_FEATURE_ORDER = [
    "Age", "Gender", "Employment_Status", "Account_Type", "Tenure_Months",
    "Number_of_Products", "Avg_Balance", "Monthly_Transactions",
    "Online_Banking_Usage", "Mobile_Banking_Usage", "Number_of_Complaints",
    "Complaint_Resolution_Rate", "Total_Digital_Usage", "Balance_per_Product"
]


def load_model():
    return joblib.load(MODEL_PATH)


def load_background():
    return pd.read_csv(BACKGROUND_PATH)


def build_input_row(age, gender, employment_status, account_type, tenure_months,
                     number_of_products, avg_balance, monthly_transactions,
                     online_banking_usage, mobile_banking_usage,
                     number_of_complaints, complaint_resolution_rate):
    """Takes the raw form inputs a user would type/select, engineers the same
    derived features used in training, and returns a single-row DataFrame in
    the exact column order the model expects."""
    total_digital_usage = online_banking_usage + mobile_banking_usage
    balance_per_product = round(avg_balance / number_of_products, 2) if number_of_products > 0 else 0.0

    row = pd.DataFrame([{
        "Age": age,
        "Gender": gender,
        "Employment_Status": employment_status,
        "Account_Type": account_type,
        "Tenure_Months": tenure_months,
        "Number_of_Products": number_of_products,
        "Avg_Balance": avg_balance,
        "Monthly_Transactions": monthly_transactions,
        "Online_Banking_Usage": online_banking_usage,
        "Mobile_Banking_Usage": mobile_banking_usage,
        "Number_of_Complaints": number_of_complaints,
        "Complaint_Resolution_Rate": complaint_resolution_rate,
        "Total_Digital_Usage": total_digital_usage,
        "Balance_per_Product": balance_per_product,
    }])[RAW_FEATURE_ORDER]
    return row


def risk_category(probability):
    if probability < 0.33:
        return "Low", "🟢"
    elif probability < 0.66:
        return "Medium", "🟡"
    else:
        return "High", "🔴"


def predict(model, input_row):
    """Returns (churn_probability, risk_label, risk_emoji)."""
    proba = model.predict_proba(input_row)[0, 1]
    label, emoji = risk_category(proba)
    return float(proba), label, emoji


def explain(model, background_df, input_row, top_n=6):
    """Returns a list of (feature_name, shap_value) for the given single input
    row, sorted by absolute contribution, using the model's own LinearExplainer
    against a precomputed background sample -- the same setup used in Task 5."""
    preprocessor = model.named_steps["prep"]
    classifier = model.named_steps["clf"]

    feature_names = preprocessor.get_feature_names_out()
    feature_names_clean = [f.split("__")[-1] for f in feature_names]

    background_t = preprocessor.transform(background_df)
    input_t = preprocessor.transform(input_row)

    explainer = shap.LinearExplainer(classifier, background_t)
    shap_values = explainer(input_t)

    contributions = list(zip(feature_names_clean, shap_values.values[0]))
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions[:top_n], shap_values, feature_names_clean, input_t
