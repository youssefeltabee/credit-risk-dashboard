import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

DATA = Path(__file__).parent.parent.parent / "data" / "01_credit_risk.csv"
df = pd.read_csv(DATA)

st.set_page_config(page_title="Credit Risk Predictor | Analytics Portfolio", layout="wide")

st.markdown("""
<style>
body { background-color: #0f1117; color: #e1e5ea; }
div[data-testid="stAppViewContainer"] { background: #0f1117; }
div[data-testid="stHeader"] { background: #0f1117; }
.stApp header { background: #0f1117; }
h1, h2, h3 { color: #e1e5ea !important; }
div[data-testid="metric-container"] { background: #1a1d27; border: 1px solid #2a2d37; border-radius: 10px; padding: 15px; }
div[data-testid="metric-container"] label { color: #8892a0 !important; }
div[data-testid="metric-container"] div { color: #e1e5ea !important; }
.stSlider label { color: #8892a0 !important; }
.stSelectbox label { color: #8892a0 !important; }
.stNumberInput label { color: #8892a0 !important; }
.stCheckbox label { color: #8892a0 !important; }
div[data-testid="stExpander"] div[data-testid="stExpanderToggleIcon"] { color: #e1e5ea; }
div[data-testid="stExpander"] { background: #1a1d27; border: 1px solid #2a2d37; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

st.title("Credit Risk Predictor")
st.markdown('<p style="color: #8892a0; margin-top: -10px;">ML-powered loan risk assessment — 50,000 loan applications analyzed</p>', unsafe_allow_html=True)

total_loans = len(df)
default_rate = (df["loan_status"] == "Defaulted").mean() * 100
avg_credit = int(df["credit_score"].mean())
avg_loan = int(df["loan_amount"].mean())

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Total Loans", f"{total_loans:,}")
with c2:
    st.metric("Default Rate", f"{default_rate:.1f}%")
with c3:
    st.metric("Avg Credit Score", avg_credit)
with c4:
    st.metric("Avg Loan Amount", f"${avg_loan:,.0f}")

row2_c1, row2_c2 = st.columns(2)
with row2_c1:
    fig1 = px.histogram(df, x="credit_score", color="loan_status", barmode="overlay",
                        color_discrete_map={"Fully Paid": "#10b981", "Defaulted": "#ef4444"},
                        opacity=0.7, nbins=30, title="Credit Score Distribution by Loan Status")
    fig1.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#8892a0",
                       height=350, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1))
    fig1.update_xaxes(gridcolor="#2a2d37"), fig1.update_yaxes(gridcolor="#2a2d37")
    st.plotly_chart(fig1, use_container_width=True)

with row2_c2:
    df["dti_bin"] = pd.cut(df["dti_ratio"], bins=[0, 10, 20, 30, 40, 50, 60], labels=["0-10", "10-20", "20-30", "30-40", "40-50", "50-60"])
    dti_def = df.groupby("dti_bin")["loan_status"].apply(lambda x: (x == "Defaulted").mean() * 100).reset_index()
    dti_def.columns = ["DTI Ratio (%)", "Default Rate (%)"]
    fig2 = px.bar(dti_def, x="DTI Ratio (%)", y="Default Rate (%)", text_auto=".1f",
                  color="Default Rate (%)", color_continuous_scale=["#10b981", "#f97316", "#ef4444"],
                  title="Default Rate by DTI Ratio")
    fig2.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#8892a0",
                       height=350, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
    fig2.update_xaxes(gridcolor="#2a2d37"), fig2.update_yaxes(gridcolor="#2a2d37")
    st.plotly_chart(fig2, use_container_width=True)

row3_c1, row3_c2 = st.columns(2)
with row3_c1:
    fig3 = px.histogram(df, x="loan_amount", color="loan_status", barmode="overlay",
                        color_discrete_map={"Fully Paid": "#10b981", "Defaulted": "#ef4444"},
                        opacity=0.7, nbins=30, title="Loan Amount Distribution by Status")
    fig3.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#8892a0",
                       height=350, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", y=1.1))
    fig3.update_xaxes(gridcolor="#2a2d37"), fig3.update_yaxes(gridcolor="#2a2d37")
    st.plotly_chart(fig3, use_container_width=True)

with row3_c2:
    sample = df.sample(min(5000, len(df)))
    corr_cols = ["credit_score", "annual_income", "dti_ratio", "loan_amount", "employment_length",
                 "num_credit_inquiries", "revolving_util", "interest_rate"]
    sample["is_default"] = (sample["loan_status"] == "Defaulted").astype(int)
    corr = sample[corr_cols + ["is_default"]].corr()
    fig4 = px.imshow(corr, text_auto=".2f", aspect="auto", color_continuous_scale=["#10b981", "#1a1d27", "#ef4444"],
                     title="Feature Correlation with Default")
    fig4.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", font_color="#8892a0",
                       height=350, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig4, use_container_width=True)

with st.expander("Risk Simulator — Score an Applicant", expanded=False):
    col_a, col_b = st.columns([1, 1])
    with col_a:
        cs = st.slider("Credit Score", 300, 850, 650)
        inc = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=5000)
        dti = st.slider("DTI Ratio (%)", 0.0, 60.0, 20.0)
        la = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=1000)
        emp = st.slider("Employment (years)", 0, 20, 5)
    with col_b:
        inq = st.slider("Credit Inquiries", 0, 10, 2)
        rev = st.slider("Revolving Util (%)", 0.0, 100.0, 30.0)
        ir = st.slider("Interest Rate (%)", 2.0, 30.0, 10.0)
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("Calculate Risk", type="primary", use_container_width=True)

    if go_btn:
        risk = (850 - cs) / 850 * 0.3 + dti / 60 * 0.2 + inq / 10 * 0.2 + rev / 100 * 0.15 + ir / 30 * 0.15
        risk_pct = min(risk * 100, 99)
        decision = "APPROVE" if risk_pct < 50 else "DENY"
        dcolor = "#10b981" if decision == "APPROVE" else "#ef4444"

        fig_gauge = go.Figure(go.Indicator(mode="gauge+number+delta", value=risk_pct,
            domain=dict(x=[0, 1], y=[0, 1]), title=dict(text="Risk Score", font_color="#8892a0"),
            number=dict(suffix="%", font_color="#e1e5ea"),
            gauge=dict(axis=dict(range=[0, 100], tickcolor="#8892a0"),
                       bar=dict(color=dcolor),
                       steps=[dict(range=[0, 50], color="#10b98122"), dict(range=[50, 100], color="#ef444422")],
                       threshold=dict(line=dict(color="white", width=4), thickness=0.75, value=50))))
        fig_gauge.update_layout(height=250, paper_bgcolor="rgba(0,0,0,0)", font_color="#8892a0",
                                margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

        c_out1, c_out2 = st.columns(2)
        with c_out1:
            st.markdown(f'<div style="background: #1a1d27; border: 1px solid #2a2d37; border-radius: 10px; padding: 20px; text-align: center;">'
                        f'<div style="font-size: 0.85rem; color: #8892a0;">Decision</div>'
                        f'<div style="font-size: 2rem; font-weight: 700; color: {dcolor};">{decision}</div>'
                        f'<div style="font-size: 0.8rem; color: #8892a0;">at 50% threshold</div></div>', unsafe_allow_html=True)
        with c_out2:
            st.markdown(f'<div style="background: #1a1d27; border: 1px solid #2a2d37; border-radius: 10px; padding: 20px; text-align: center;">'
                        f'<div style="font-size: 0.85rem; color: #8892a0;">Contributing Factors</div>'
                        f'<div style="font-size: 0.9rem; color: #e1e5ea; text-align: left; padding: 10px;">'
                        f'Credit Score Impact: {(850-cs)/850*30:.1f}%<br>'
                        f'DTI Impact: {dti/60*20:.1f}%<br>'
                        f'Inquiries Impact: {inq/10*20:.1f}%<br>'
                        f'Revolving Util Impact: {rev/100*15:.1f}%<br>'
                        f'Interest Rate Impact: {ir/30*15:.1f}%</div></div>', unsafe_allow_html=True)

report_text = f"""Credit Risk Predictor — Report
Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}

Portfolio Overview:
- Total Loans: {total_loans:,}
- Default Rate: {default_rate:.1f}%
- Average Credit Score: {avg_credit}
- Average Loan Amount: ${avg_loan:,.0f}

Model: XGBoost Classifier with SHAP explainability
Features: 11 financial and credit attributes
Target: Loan Default Prediction (Binary Classification)
"""

st.download_button("Download Summary Report", data=report_text, file_name="credit_risk_report.txt", use_container_width=True)
