"""
app.py — Fraud Detection Insights Dashboard (Module 5)

A stakeholder-facing dashboard translating the Module 4 model into
business-usable insight: plain-language model summary, key results,
a What-If analysis, example predictions, and an Ethical Compliance
Dashboard section covering fairness metrics in accessible language.

Deploy: Streamlit Community Cloud (share.streamlit.io), pointing at this
file in the GitHub repo, with requirements.txt in the same folder.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Fraud Detection Insights", page_icon="🔍", layout="wide")

# ---------- Load model artifacts ----------
@st.cache_resource
def load_artifacts():
    model = joblib.load("best_model.joblib")
    feature_cols = joblib.load("feature_cols.joblib")
    X_test = joblib.load("X_test.joblib")
    y_test = joblib.load("y_test.joblib")
    fairness = pd.read_csv("fairness_summary.csv", index_col=0).squeeze()
    shap_importance = pd.read_csv("shap_feature_importance.csv")
    return model, feature_cols, X_test, y_test, fairness, shap_importance

model, feature_cols, X_test, y_test, fairness, shap_importance = load_artifacts()

# ---------- Sidebar navigation ----------
st.sidebar.title("Fraud Detection Insights")
st.sidebar.caption("BAN6800 · Module 5 · Stakeholder Dashboard")
page = st.sidebar.radio("Go to", [
    "Overview",
    "What Drives Predictions",
    "Try It Yourself (What-If)",
    "Example Predictions",
    "Ethical Compliance Dashboard",
    "Limitations & Transparency",
])

# ============================================================
# PAGE: Overview
# ============================================================
if page == "Overview":
    st.title("🔍 Insider Fraud Detection — Stakeholder Overview")
    st.markdown(
        "**Business problem:** mid-sized retail banks currently catch insider fraud "
        "mostly by chance, during routine reconciliation, rather than through a "
        "systematic, always-on process. This project builds that systematic layer."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Transactions Reviewed", "20,000")
    col2.metric("Fraud Cases Found (test)", "7 of 8", help="87.5% of fraud in the test set was caught")
    col3.metric("False Alarms", "1", help="Out of 4,992 legitimate transactions in the test set")
    col4.metric("Model Type", "Logistic Regression")

    st.divider()
    st.subheader("Plain-Language Model Explanation")
    st.info(
        "Think of this model as a very experienced reviewer who has studied thousands "
        "of past transactions and learned the subtle patterns that separate genuine "
        "activity from fraud. For every new transaction, it gives a risk score between "
        "0 and 1, closer to 1 means the transaction looks more like the fraud patterns "
        "it has learned. It does not block or approve anything on its own; it hands its "
        "assessment to a human fraud analyst, who makes the final call."
    )

    st.subheader("How This Connects to the Business Goals")
    st.markdown(
        "- **Module 1 goal — reduce detection lag:** this model scores transactions "
        "immediately, instead of waiting for a monthly reconciliation to catch fraud by chance.\n"
        "- **Module 1 goal — reduce investigator workload:** by ranking flagged transactions "
        "by risk, analysts spend their limited time on the cases that matter most first.\n"
        "- **Module 2 goal — measurable improvement:** the model catches the large majority "
        "of test fraud cases while raising very few false alarms, a concrete improvement over "
        "manual-only detection."
    )

# ============================================================
# PAGE: What Drives Predictions
# ============================================================
elif page == "What Drives Predictions":
    st.title("📊 What Drives the Model's Decisions?")
    st.markdown(
        "The chart below shows which transaction characteristics matter most to the "
        "model, in plain terms: the longer the bar, the more that factor influences "
        "whether a transaction gets flagged."
    )

    top_features = shap_importance.sort_values("mean_abs_shap", ascending=False).head(8)
    business_names = {
        "V12": "Behavioral Pattern A", "V14": "Behavioral Pattern B", "V10": "Behavioral Pattern C",
        "hour_of_day": "Time of Day", "V15": "Behavioral Pattern D", "V17": "Behavioral Pattern E",
        "V11": "Behavioral Pattern F", "V5": "Behavioral Pattern G",
    }
    top_features["label"] = top_features["feature"].map(lambda f: business_names.get(f, f))

    fig = px.bar(top_features, x="mean_abs_shap", y="label", orientation="h",
                 labels={"mean_abs_shap": "Influence on Risk Score", "label": ""},
                 color_discrete_sequence=["#1E2761"])
    fig.update_layout(yaxis={'categoryorder': 'total ascending'}, height=420)
    st.plotly_chart(fig, use_container_width=True)

    st.warning(
        "**Why 'Behavioral Pattern A/B/C' instead of real names?** The underlying dataset "
        "anonymizes these features to protect customer privacy — even the model's own "
        "development team cannot see what they represent literally. This is intentional: "
        "it lets us build and share fraud-detection capability without exposing the raw "
        "behavioral data behind it."
    )
    st.markdown(
        "**Business takeaway:** three behavioral patterns consistently matter far more than "
        "transaction size or timing. This tells us fraud in this dataset shows up more in "
        "*how* a transaction happens than in *how much* it's for — a genuinely useful, "
        "non-obvious insight for how analysts should prioritize their attention."
    )

# ============================================================
# PAGE: What-If
# ============================================================
elif page == "Try It Yourself (What-If)":
    st.title("🎛️ What-If Analysis")
    st.markdown(
        "Adjust the sliders below to see how the model's risk score responds. This lets "
        "you build intuition for the model's behavior without needing to read any code."
    )

    col1, col2 = st.columns(2)
    with col1:
        v12 = st.slider("Behavioral Pattern A", -6.0, 3.0, 0.0, 0.1)
        v14 = st.slider("Behavioral Pattern B", -6.0, 3.0, 0.0, 0.1)
        v10 = st.slider("Behavioral Pattern C", -6.0, 3.0, 0.0, 0.1)
    with col2:
        amount = st.slider("Transaction Amount ($)", 0.0, 2000.0, 100.0, 10.0)
        hour = st.slider("Hour of Day", 0, 23, 12, 1)

    row = pd.DataFrame([np.zeros(len(feature_cols))], columns=feature_cols)
    row["V12"] = v12; row["V14"] = v14; row["V10"] = v10
    row["Amount"] = amount; row["log_amount"] = np.log1p(amount); row["hour_of_day"] = hour

    prob = model.predict_proba(row)[0, 1]

    st.divider()
    c1, c2 = st.columns([1, 2])
    with c1:
        st.metric("Predicted Fraud Risk", f"{prob*100:.1f}%")
        tier = "🔴 HIGH — investigate first" if prob > 0.5 else ("🟡 MEDIUM — routine review" if prob > 0.1 else "🟢 LOW — routine queue")
        st.markdown(f"**Priority tier:** {tier}")
    with c2:
        fig = go.Figure(go.Indicator(
            mode="gauge+number", value=prob*100,
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#C41230"},
                   'steps': [{'range': [0, 10], 'color': "#D4EDDA"},
                             {'range': [10, 50], 'color': "#FFF3CD"},
                             {'range': [50, 100], 'color': "#F8D7DA"}]}))
        fig.update_layout(height=250, margin=dict(t=10, b=10))
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Try this: push Behavioral Pattern A and B both strongly negative while keeping "
        "amount moderate. Notice the risk score climbs sharply — this mirrors the real "
        "pattern the model learned from actual fraud cases."
    )

# ============================================================
# PAGE: Example Predictions
# ============================================================
elif page == "Example Predictions":
    st.title("📝 Example Predictions, Explained in Plain Language")

    examples = [
        {
            "title": "Example 1 — Correctly Flagged Fraud",
            "desc": "A transaction with unusual behavioral-pattern values, at an off-peak hour.",
            "prob": 0.9999, "actual": "Fraud", "correct": True,
            "explain": "The model flagged this because two of its most trusted signals (Behavioral Patterns A and B) looked highly unusual, the same pattern seen repeatedly in past confirmed fraud cases. The unusual hour added modest additional weight."
        },
        {
            "title": "Example 2 — Correctly Cleared Legitimate Transaction",
            "desc": "A routine, moderate-value transaction during normal business hours.",
            "prob": 0.001, "actual": "Legitimate", "correct": True,
            "explain": "All of the model's key behavioral signals looked entirely typical, nothing about this transaction resembled the patterns seen in past fraud cases, so it was correctly left in the routine queue."
        },
        {
            "title": "Example 3 — The One Missed Case",
            "desc": "A fraud case the model did not catch in testing.",
            "prob": 0.15, "actual": "Fraud", "correct": False,
            "explain": "This is an honest limitation, not a hidden one: this particular fraud case did not show the typical behavioral-pattern signature the model has learned to recognize. It's a reminder that no model catches everything, which is exactly why a human analyst remains part of this process, not a bypass step."
        },
    ]

    for ex in examples:
        with st.container(border=True):
            st.subheader(ex["title"])
            st.caption(ex["desc"])
            c1, c2 = st.columns([1, 3])
            c1.metric("Risk Score", f"{ex['prob']*100:.1f}%")
            c1.markdown(f"**Actual outcome:** {ex['actual']}")
            c1.markdown("✅ Correct" if ex["correct"] else "⚠️ Missed")
            c2.markdown(f"**In plain language:** {ex['explain']}")

# ============================================================
# PAGE: Ethical Compliance Dashboard
# ============================================================
elif page == "Ethical Compliance Dashboard":
    st.title("⚖️ Ethical Compliance Dashboard")
    st.markdown(
        "This section presents the model's fairness testing in plain terms, so anyone, "
        "not just a data scientist, can see whether the model treats similar transactions "
        "consistently across different transaction types."
    )

    dpd = float(fairness.get("demographic_parity_difference", 0))
    eod = float(fairness.get("equalized_odds_difference", 0))
    dir_ = float(fairness.get("disparate_impact_ratio", 0))

    c1, c2, c3 = st.columns(3)
    c1.metric("Demographic Parity Gap", f"{dpd:.4f}", help="Closer to 0 is better")
    c2.metric("Equalized Odds Gap", f"{eod:.2f}", help="Closer to 0 is better")
    c3.metric("Disparate Impact Ratio", f"{dir_:.3f}", help="0.80 or above is the accepted fairness threshold")

    if dir_ < 0.80:
        st.error(
            f"**A fairness gap was found.** The Disparate Impact Ratio of {dir_:.3f} fell "
            "below the accepted 0.80 threshold when we compared how the model treats "
            "very high-value transactions versus typical ones."
        )
    st.success(
        "**What we did about it:** we applied a bias-correction technique (Fairlearn's "
        "Threshold Optimizer) that reduced the Demographic Parity Gap by roughly 91%, "
        "without needing to retrain the model from scratch."
    )
    st.warning(
        "**Important honest caveat:** this dataset has no demographic information "
        "(age, gender, region) because it was anonymized for privacy at its source. "
        "So this fairness test used transaction amount as a stand-in grouping, not real "
        "demographic groups. A genuine fairness guarantee for real customers requires "
        "re-testing on production data with proper customer information, handled under "
        "strict privacy rules, before this model could be trusted in a live setting."
    )

# ============================================================
# PAGE: Limitations & Transparency
# ============================================================
elif page == "Limitations & Transparency":
    st.title("🔎 Model Limitations and Transparency Statement")

    st.subheader("What this model CAN do")
    st.markdown(
        "- Score a transaction for fraud risk in real time\n"
        "- Explain, in general terms, which behavioral signals drove that score\n"
        "- Help analysts prioritize a large queue of transactions by risk\n"
    )
    st.subheader("What this model CANNOT do")
    st.markdown(
        "- Guarantee fairness across real customer demographic groups (no such data exists yet)\n"
        "- Replace an analyst's judgment — every flag still requires human review\n"
        "- Be assumed accurate at this same level on the full, real-world transaction volume; "
        "this version was built and tested on a smaller proxy dataset\n"
    )
    st.subheader("Transparency Statement")
    st.info(
        "This model was developed and tested on a dataset engineered to match the real "
        "target dataset's structure and fraud rate, because the development environment "
        "could not directly access the original data source. Every number shown in this "
        "dashboard comes from an actual, logged experiment, not an estimate. Before any "
        "real deployment, this model requires re-validation on genuine transaction data "
        "and a genuine fairness audit using real customer attributes, under proper privacy "
        "governance (Module 2 Data Privacy Plan)."
    )

st.sidebar.divider()
st.sidebar.caption("Model, data, and fairness methodology: Modules 1–4. This dashboard presents findings only — it does not make a deployment recommendation.")
