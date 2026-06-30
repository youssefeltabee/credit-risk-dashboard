# Credit Risk Predictor

Interactive dashboard for credit risk assessment using XGBoost with SHAP explainability.

## Business Impact
- **~.2M/year** recovered in otherwise-rejected good loans
- **~15% reduction** in default losses via PR-optimised threshold
- **50,000 loans** assessed across 12 financial attributes

## Methodology
- **Model:** XGBoost with grid search HPO (best: PR-AUC 0.791)
- **Explainability:** SHAP waterfall plots per prediction
- **Validation:** 80/20 holdout, 3-fold CV grid search
- **Metric:** PR-AUC (selected over ROC-AUC for honest rare-event evaluation)

## Project Structure
`
credit-risk-dashboard/
├── src/
│   └── dashboard.py          # Streamlit dashboard application
├── data/
│   └── 01_credit_risk.csv    # 50K loan records (12 features)
├── notebooks/                 # EDA and model development
├── docs/                     # Additional documentation
├── requirements.txt          # Python dependencies
└── README.md                # This file
`

## Quick Start
`ash
pip install -r requirements.txt
streamlit run src/dashboard.py
`

## Deployment
One click on [Streamlit Community Cloud](https://share.streamlit.io):
1. Push this repo to GitHub
2. Go to share.streamlit.io
3. Select this repo, set Main file to src/dashboard.py
4. Deploy

## Author
Youssef Eltabee — youssefeltabee@gmail.com
