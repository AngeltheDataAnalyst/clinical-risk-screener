import json
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. Load cleaned dataset
df = pd.read_csv("data/clinical_risk.csv")
X = df.drop(columns=["high_risk"])
y = df["high_risk"]

# 2. Train / Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Define candidate models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42),
}

best_name = None
best_model = None
best_auc = 0.0
results = {}

print("=== Model Comparison on Test Set ===")
for name, clf in models.items():
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    results[name] = {"accuracy": round(acc, 4), "roc_auc": round(auc, 4)}

    print(f"\n{name}:")
    print(f"  Accuracy : {acc:.2%}")
    print(f"  ROC-AUC  : {auc:.4f}")

    if auc > best_auc:
        best_auc = auc
        best_name = name
        best_model = clf

# 4. Save winning model and metadata
joblib.dump(best_model, "model/risk_model.pkl")

metadata = {
    "best_model": best_name,
    "features": list(X.columns),
    "metrics": results,
    "feature_ranges": {
        col: {
            "min": float(X[col].min()),
            "max": float(X[col].max()),
            "mean": float(X[col].mean()),
        }
        for col in X.columns
    },
}

with open("model/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"\nSaved winning model ({best_name}) to model/risk_model.pkl!")