import json
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Clinical Risk Screener",
    page_icon="🩺",
    layout="wide",
)

@st.cache_resource
def load_artifacts():
    model = joblib.load("model/risk_model.pkl")
    with open("model/metadata.json", "r") as f:
        meta = json.load(f)
    df = pd.read_csv("data/clinical_risk.csv")
    return model, meta, df

model, meta, df = load_artifacts()
features = meta["features"]
ranges = meta["feature_ranges"]

FEATURE_LABELS = {
    "age": "Age (Standardized)",
    "sex": "Sex (Standardized)",
    "bmi": "Body Mass Index (BMI)",
    "blood_pressure": "Mean Blood Pressure",
    "tc_cholesterol": "Total Serum Cholesterol (TC)",
    "ldl": "Low-Density Lipoproteins (LDL)",
    "hdl": "High-Density Lipoproteins (HDL)",
    "tch": "Total Cholesterol / HDL Ratio",
    "ltg_triglycerides": "Serum Triglycerides (Log)",
    "glucose": "Blood Glucose Level",
}

st.title("🩺 Early Metabolic & Clinical Risk Screener")
st.markdown(
    "An interactive machine learning screening tool built with **Python, Scikit-learn, and Streamlit** "
    "to estimate elevated disease progression risk from standardized clinical biomarkers."
)
st.info(
    "**Educational Disclaimer:** This prototype is built for data science demonstration and educational "
    "purposes only. It is **not** a medical device and should not replace professional clinical advice."
)

# Sidebar Inputs
st.sidebar.header("Patient Biomarker Inputs")
st.sidebar.caption("Adjust standardized clinical values below (0.0 = population average):")

preset = st.sidebar.radio(
    "Quick Sample Presets",
    ["Custom / Average", "Sample Lower-Risk Profile", "Sample Elevated-Risk Profile"],
)

if preset == "Sample Lower-Risk Profile":
    sample_row = df[df["high_risk"] == 0].head(1).squeeze()
elif preset == "Sample Elevated-Risk Profile":
    sample_row = df[df["high_risk"] == 1].head(1).squeeze()
else:
    sample_row = None

user_inputs = {}
for feat in features:
    f_min = float(np.floor(ranges[feat]["min"] * 100) / 100)
    f_max = float(np.ceil(ranges[feat]["max"] * 100) / 100)
    default_val = float(sample_row[feat]) if sample_row is not None else float(round(ranges[feat]["mean"], 3))
    user_inputs[feat] = st.sidebar.slider(
        label=FEATURE_LABELS.get(feat, feat),
        min_value=f_min,
        max_value=f_max,
        value=float(np.clip(default_val, f_min, f_max)),
        step=0.005,
    )

input_df = pd.DataFrame([user_inputs])

# Prediction
probs = model.predict_proba(input_df).ravel()
prob_high = float(probs.item(1))
risk_pct = prob_high * 100

if prob_high >= 0.65:
    risk_tier = "Elevated Risk"
    badge_color = "#d9534f"
elif prob_high >= 0.35:
    risk_tier = "Moderate Risk"
    badge_color = "#f0ad4e"
else:
    risk_tier = "Lower Risk"
    badge_color = "#5cb85c"

# Main Dashboard Tabs
tab1, tab2, tab3 = st.tabs(["Risk Assessment", "Model Performance", "Dataset Explorer"])

with tab1:
    col1, col2 = st.columns(list((1, 1.3)))

    with col1:
        st.subheader("Predicted Risk Score")
        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                number={"suffix": "%", "font": {"size": 42}},
                title={"text": f"Tier: {risk_tier}", "font": {"size": 20}},
                gauge={
                    "axis": {"range": list((0, 100))},
                    "bar": {"color": badge_color},
                    "steps": [
                        {"range": list((0, 35)), "color": "#e8f5e9"},
                        {"range": list((35, 65)), "color": "#fff8e1"},
                        {"range": list((65, 100)), "color": "#ffebee"},
                    ],
                },
            )
        )
        fig_gauge.update_layout(height=320, margin=dict(l=20, r=20, t=50, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col2:
        st.subheader("What Drives This Prediction? (Feature Impact)")
        if hasattr(model, "coef_"):
            coefs = model.coef_.ravel()
            vals = input_df.to_numpy().ravel()
            contributions = coefs * vals
            contrib_df = pd.DataFrame(
                {
                    "Biomarker": [FEATURE_LABELS.get(f, f) for f in features],
                    "Risk Impact": contributions,
                }
            ).sort_values("Risk Impact")

            fig_bar = px.bar(
                contrib_df,
                x="Risk Impact",
                y="Biomarker",
                orientation="h",
                color="Risk Impact",
                color_continuous_scale=["#2e7d32", "#eeeeee", "#c62828"],
                title="Positive values push risk higher; negative values lower risk",
            )
            fig_bar.update_layout(height=340, margin=dict(l=10, r=10, t=40, b=10), coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Model Benchmarking Results")
    st.write(f"**Selected Production Model:** `{meta['best_model']}` (chosen by highest ROC-AUC on held-out test set)")
    metrics_df = pd.DataFrame(meta["metrics"]).T.reset_index().rename(columns={"index": "Model"})
    st.dataframe(metrics_df, use_container_width=True)

with tab3:
    st.subheader("Clinical Dataset Overview (442 Patients)")
    st.dataframe(df.head(20), use_container_width=True)
    fig_hist = px.histogram(
        df,
        x="bmi",
        color=df["high_risk"].map({0: "Lower Risk (0)", 1: "Elevated Risk (1)"}),
        barmode="overlay",
        title="BMI Distribution by Risk Group",
        labels={"color": "Risk Group", "bmi": "Standardized BMI"},
    )
    st.plotly_chart(fig_hist, use_container_width=True)