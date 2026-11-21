# 🩺 Early Metabolic & Clinical Risk Screener

An interactive machine learning screening tool built with **Python, Scikit-learn, Plotly, and Streamlit** to estimate elevated disease progression risk from standardized clinical biomarkers and visualize individual feature contributions.

> **Educational Disclaimer:** This prototype is built for data science demonstration and educational purposes only. It is not a medical device and does not provide clinical diagnoses.

## Key Features
- **Real-Time Risk Assessment:** Interactive sidebar sliders and sample patient presets with an instant probability gauge and risk tier indicator.
- **Explainable Predictions:** Horizontal feature-impact bar chart showing which biomarkers raise or lower an individual's predicted risk.
- **Model Benchmarking:** Automated comparison of **Logistic Regression** (Accuracy: `74.16%`, ROC-AUC: `0.8380`) and **Random Forest** (Accuracy: `76.40%`, ROC-AUC: `0.8228`), selecting the top ROC-AUC model for inference.
- **Dataset Explorer:** Interactive exploration of the 442-patient clinical dataset and biomarker distributions.

## Tech Stack
- **Language:** Python
- **Data & Modeling:** Pandas, NumPy, Scikit-learn, Joblib
- **Frontend & Visualization:** Streamlit, Plotly

## How to Run Locally
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate dataset & train model:
   ```bash
   python get_data.py
   python train_model.py
   ```
3. Launch the Streamlit app:
   ```bash
   streamlit run app.py
   ```