# CDAM_09-26_IP028_CustomerChurnPrediction_Kinyua# Customer Churn Prediction & Risk-Scoring Model

A machine learning project that predicts which financial-institution customers are at risk of churning, explains *why* using Explainable AI, and deploys the model as a live, interactive risk-scoring tool.

**🔗 Live app:** https://customerchurnnprediction.streamlit.app/
---

## Business Problem

Financial institutions lose deposits, transaction revenue, and cross-selling opportunities every time a customer churns. This project uses historical customer data (demographics, account activity, digital-banking usage, complaints, and more) to answer six strategic questions:

1. **What** is the current churn rate and pattern?
2. **Who** is most likely to churn?
3. **Why** do customers churn?
4. **When** are customers most vulnerable?
5. **Which** segments carry the highest risk?
6. **How** can predictive analytics support retention?

## Key Findings

- **Churn rate:** ~26% of established customers (tenure > 12 months) have churned — a meaningful class imbalance, handled throughout with appropriate metrics (ROC-AUC, F1, Recall) rather than accuracy alone.
- **Tenure is the dominant driver of churn risk** — confirmed independently across EDA correlations, SHAP global importance, and Partial Dependence Plots. It's roughly **5.5x** more influential than the next-strongest factor.
- **Complaint history is the second-most influential, and actionable, factor** — unlike tenure, this is something a retention team can act on directly.
- **Balance, digital usage, account type, and demographics are weak standalone predictors** — an honest, data-supported finding that shaped the choice of a simpler, more interpretable final model.
- **Final model:** Logistic Regression, selected from 5 tuned algorithms (Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM), achieving ROC-AUC ≈ 0.62.S

## Project Structure

```
├── app/                          # Streamlit deployment
│   ├── app.py                    # UI — form, prediction display, SHAP chart
│   ├── churn_logic.py            # Prediction & explanation logic (no UI code)
│   ├── best_model.pkl            # Trained model (local copy for the app)
│   └── background_sample.csv     # Reference sample for SHAP explanations
│
├── notebooks/
│   ├── data_cleaning_and_wrangling.ipynb        # Tasks 1-2: import, clean, wrangle
│   ├── Exploratory_Data_Analysis.ipynb          # Task 3: exploratory visualization
│   ├── Machine_Learning.ipynb                   # Task 4: train, tune, compare 5 models
│   └── Explainable_AI                           # Task 5: SHAP, PDP, individual explanations
│
├── data/
│                       # Original untouched dataset Cleaned, wrangled, and train/test split data
│                     
│
├── models/
│   └── best_model.pkl            # Master copy of the trained model
│
├── requirements.txt              # Exact package versions used
└── README.md
```

## Methodology

| Stage | What was done |
|---|---|
| **Data Cleaning** | Handled missing `Education_Level` values, flagged a `Customer_ID` reuse data-quality issue, checked for unrealistic values |
| **Data Wrangling** | Filtered to established customers, engineered `Total_Digital_Usage` and `Balance_per_Product`, grouped and summarized by segment |
| **EDA** | 8 chart types (histogram, boxplot, scatter, bar, pie, heatmap, pair plot, violin), each with interpretation |
| **Modeling** | 5 algorithms, `RandomizedSearchCV` hyperparameter tuning, 5-fold stratified cross-validation, evaluated on Accuracy, Precision, Recall, Specificity, F1, ROC-AUC, PR-AUC, and Cohen's Kappa |
| **Explainability** | SHAP (global importance + individual waterfall breakdowns), Partial Dependence Plots |
| **Deployment** | Streamlit app with real-time predictions, risk categorization, and live SHAP-based explanations |

## Model Performance

| Model | ROC-AUC | F1 | Recall | Accuracy |
|---|---|---|---|---|
| **Logistic Regression** ✅ | **0.622** | 0.426 | 0.591 | 0.579 |
| XGBoost | 0.617 | 0.421 | 0.569 | 0.586 |
| LightGBM | 0.617 | 0.417 | 0.482 | 0.644 |
| Random Forest | 0.616 | 0.428 | 0.581 | 0.590 |
| Decision Tree | 0.564 | 0.405 | 0.603 | 0.531 |

*Note on interpreting these numbers: performance is modest across all 5 models, consistent with genuinely weak correlations found in EDA. The model is best used as a risk-prioritization tool, not a high-confidence predictor.*

## Tech Stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · matplotlib/seaborn · Streamlit

## Running Locally

```bash
# Clone the repo
git clone https://github.com/VictorMfana/CDAM_09-26_IP028_CustomerChurnPrediction_Kinyua.git
cd CDAM_09-26_IP028_CustomerChurnPrediction_Kinyua

# Set up environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run the notebooks
jupyter notebook notebooks/

# Run the app
streamlit run app/app.py
```

## Author - Victor Mfana Kinyua


Built as part of a Machine Learning internship project on customer churn prediction at the Center for data analytics and Modelling at Chuka University
