import pandas as pd
from sklearn.datasets import load_diabetes

# Load built-in clinical diabetes dataset (442 patients, 10 clinical features)
data = load_diabetes(as_frame=True)
df = data.frame.copy()

# Create a clear binary risk target (1 = Elevated Progression Risk, 0 = Lower Risk)
median_val = df["target"].median()
df["high_risk"] = (df["target"] > median_val).astype(int)
df = df.drop(columns=["target"])

# Rename columns to clear clinical names
df.columns = [
    "age", "sex", "bmi", "blood_pressure",
    "tc_cholesterol", "ldl", "hdl", "tch", "ltg_triglycerides", "glucose",
    "high_risk"
]

df.to_csv("data/clinical_risk.csv", index=False)
print("Dataset saved successfully! Shape:", df.shape)
print(df.head())