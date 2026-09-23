"""
streamlit_app.py
Customer Churn Risk-Scoring App — Task 6 deployment.
Run locally with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os

from churn_logic import load_model, load_background, build_input_row, predict, explain

st.set_page_config(page_title="Customer Churn Risk Scoring", page_icon="📊", layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
COMPARISON_TABLE_PATH = os.path.join(BASE_DIR, "..", "data", "model_comparison_table.csv")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_background():
    return load_background()


@st.cache_data
def get_comparison_table():
    try:
        return pd.read_csv(COMPARISON_TABLE_PATH)
    except FileNotFoundError:
        return None


model = get_model()
background = get_background()

st.title("📊 Customer Churn Risk-Scoring Tool")
st.markdown(
    "Enter a customer's details to get a real-time churn risk score, "
    "along with an explanation of what's driving that score."
)

tab_predict, tab_about = st.tabs(["🔮 Predict", "ℹ️ About this model"])

# ============================================================
# TAB 1: PREDICTION
# ============================================================
with tab_predict:
    st.subheader("Customer details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        employment_status = st.selectbox("Employment Status", ["Employed", "Self-employed", "Unemployed"])
        account_type = st.selectbox("Account Type", ["Current", "Savings", "Fixed Deposit"])

    with col2:
        tenure_months = st.number_input("Account Tenure (months)", min_value=0, max_value=600, value=36)
        number_of_products = st.number_input("Number of Products", min_value=1, max_value=10, value=2)
        avg_balance = st.number_input("Average Balance", min_value=0, value=50000, step=1000)
        monthly_transactions = st.number_input("Monthly Transactions", min_value=0, max_value=200, value=10)

    with col3:
        online_banking_usage = st.number_input("Online Banking Usage (monthly logins)", min_value=0, max_value=100, value=5)
        mobile_banking_usage = st.number_input("Mobile Banking Usage (monthly logins)", min_value=0, max_value=100, value=8)
        number_of_complaints = st.number_input("Number of Complaints (last 12 months)", min_value=0, max_value=20, value=0)
        complaint_resolution_rate = st.slider("Complaint Resolution Rate", min_value=0.0, max_value=1.0, value=0.75, step=0.01)

    predict_clicked = st.button("Predict Churn Risk", type="primary")

    if predict_clicked:
        input_row = build_input_row(
            age, gender, employment_status, account_type, tenure_months,
            number_of_products, avg_balance, monthly_transactions,
            online_banking_usage, mobile_banking_usage,
            number_of_complaints, complaint_resolution_rate
        )

        proba, label, emoji = predict(model, input_row)

        st.markdown("---")
        st.subheader("Result")

        res_col1, res_col2 = st.columns([1, 2])

        with res_col1:
            st.metric("Churn Probability", f"{proba * 100:.1f}%")
            st.markdown(f"### Risk Category: {emoji} **{label}**")
            st.progress(min(proba, 1.0))

        with res_col2:
            contributions, shap_values, feature_names, input_t = explain(model, background, input_row)

            st.markdown("**What's driving this score:**")
            summary_lines = []
            for name, val in contributions:
                direction = "increases" if val > 0 else "decreases"
                summary_lines.append(f"- `{name}` **{direction}** churn risk (effect: {val:+.2f})")
            st.markdown("\n".join(summary_lines))

            #  auto-generated summary (built-in explanation assistant --
            
            top_feature, top_val = contributions[0]
            direction_word = "the biggest risk factor" if top_val > 0 else "the strongest reason this customer looks safe"
            st.info(
                f" `{top_feature}` is {direction_word} for this "
                f"customer, contributing more to the prediction than any other factor."
            )

        st.markdown("**Feature contribution breakdown**")
        fig, ax = plt.subplots(figsize=(8, 4))
        names = [c[0] for c in contributions][::-1]
        values = [c[1] for c in contributions][::-1]
        colors = ["#C44E52" if v > 0 else "#4C72B0" for v in values]
        ax.barh(names, values, color=colors)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value (log-odds contribution)")
        ax.set_title("Top factors for this prediction")
        st.pyplot(fig)

        with st.expander("What do these numbers mean?"):
            st.markdown(
                "Each bar shows how much that feature pushed this specific prediction "
                "up (red, toward churn) or down (blue, toward retention), in the model's "
                "own decision units. This is the same SHAP method used to explain the "
                "model in the project's Explainable AI stage -- every number here is "
                "traceable back to the model's actual math, not a generic rule of thumb."
            )
    else:
        st.info("Fill in the customer's details above and click **Predict Churn Risk**.")

# ============================================================
# TAB 2: ABOUT / MODEL INFO
# ============================================================
with tab_about:
    st.subheader("About this model")
    st.markdown(
        """
        This tool is powered by a **Logistic Regression** model, selected as the
        best-performing option after comparing 5 algorithms (Logistic Regression,
        Decision Tree, Random Forest, XGBoost, LightGBM) with hyperparameter tuning.

        **Important context for interpreting results:**
        - The model achieves a ROC-AUC of roughly **0.62** on held-out test data --
          a real, above-chance ability to rank customers by risk, but not a highly
          confident predictor. Treat scores as a **prioritization tool**, not a
          certainty.
        - The single strongest driver of predicted risk is **account tenure** --
          customers earlier in their relationship with the institution consistently
          score higher risk, regardless of other factors.
        - Complaint history is the next most influential factor, and it's one the
          institution can actually act on (faster resolution, proactive outreach).
        """
    )

    comparison_table = get_comparison_table()
    if comparison_table is not None:
        st.markdown("**Model comparison (from training):**")
        st.dataframe(comparison_table.set_index(comparison_table.columns[0]))

    st.caption(
        "Built for the Customer Churn Prediction internship project. "
        "Model, EDA, and explainability work documented in the project's GitHub repository."
    )
