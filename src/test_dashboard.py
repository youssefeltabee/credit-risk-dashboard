"""Tests for credit risk dashboard — validates data, model, and pipeline."""
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, average_precision_score
from xgboost import XGBClassifier

DATA = Path(__file__).parent.parent / "data" / "01_credit_risk.csv"

def test_data_shape():
    df = pd.read_csv(DATA)
    assert df.shape == (50001, 13)

def test_default_rate_range():
    df = pd.read_csv(DATA)
    rate = (df["loan_status"].str.startswith("Default")).mean()
    assert 0.20 < rate < 0.30

def test_all_features_present():
    df = pd.read_csv(DATA)
    expected = {"credit_score", "annual_income", "dti_ratio", "loan_amount",
                "loan_term", "employment_length", "num_credit_lines",
                "delinquent_history", "num_delinquencies", "num_credit_inquiries",
                "revolving_util", "interest_rate", "loan_status"}
    assert expected.issubset(set(df.columns))

def test_no_null_values():
    df = pd.read_csv(DATA)
    assert df.isnull().sum().sum() == 0

def test_model_trains_and_predicts():
    df = pd.read_csv(DATA)
    feats = ["credit_score","annual_income","dti_ratio","loan_amount","loan_term",
               "employment_length","num_credit_lines","delinquent_history",
              "num_delinquencies","num_credit_inquiries","revolving_util","interest_rate"]
    X, y = df[feats], (df["loan_status"].str.startswith("Default")).astype(int)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    m = XGBClassifier(n_estimators=50, max_depth=3, random_state=42, verbosity=0)
    m.fit(X_tr, y_tr)
    y_prob = m.predict_proba(X_te)[:, 1]
    ap = average_precision_score(y_te, y_prob)
    assert ap > 0.7, f"PR-AUC too low: {ap:.3f}"
