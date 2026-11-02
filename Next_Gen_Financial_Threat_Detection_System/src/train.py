from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from src.data_utils import prepare_dataframe

MODEL_PATH = Path("models/model_bundle.joblib")
NUMERIC = ["amount_inr","hour","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest",
           "balance_delta_orig","balance_delta_dest","is_night","amount_to_balance_ratio"]
CATEGORICAL = ["type","location","currency"]
FEATURES = NUMERIC + CATEGORICAL

def _pipeline(model):
    prep = ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL)
    ])
    return Pipeline([("preprocess", prep), ("classifier", model)])

def train_models(raw_df):
    df = prepare_dataframe(raw_df)
    if "isFraud" not in df.columns:
        raise ValueError("Training data must contain an 'isFraud' target column (0=legitimate, 1=fraud).")
    y = pd.to_numeric(df["isFraud"], errors="coerce").fillna(0).astype(int)
    if y.nunique() < 2:
        raise ValueError("The target column must contain both classes 0 and 1.")
    X = df[FEATURES]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=stratify
    )
    models = {
        "Logistic Regression": _pipeline(LogisticRegression(max_iter=1000, class_weight="balanced")),
        "Random Forest": _pipeline(RandomForestClassifier(n_estimators=180, max_depth=14,
                         min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
    }
    metrics, fitted = {}, {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:,1]
        metrics[name] = {
            "accuracy": round(accuracy_score(y_test,pred),4),
            "precision": round(precision_score(y_test,pred,zero_division=0),4),
            "recall": round(recall_score(y_test,pred,zero_division=0),4),
            "f1": round(f1_score(y_test,pred,zero_division=0),4),
            "roc_auc": round(roc_auc_score(y_test,proba),4) if y_test.nunique()==2 else 0.0
        }
        fitted[name] = model
    best_name = max(metrics, key=lambda k: metrics[k]["f1"])
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fitted[best_name], "model_name": best_name, "features": FEATURES,
                 "metrics": metrics}, MODEL_PATH)
    return metrics

def load_model_bundle():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No model file. Train a model first.")
    return joblib.load(MODEL_PATH)

def predict_dataframe(raw_df, bundle=None):
    if bundle is None:
        bundle = load_model_bundle()
    df = prepare_dataframe(raw_df)
    probs = bundle["model"].predict_proba(df[FEATURES])[:,1]
    # Default threshold 0.50; adjust after validation for the institution's risk appetite.
    out = df.copy()
    out["fraud_probability"] = probs
    out["prediction"] = (probs >= 0.50).astype(int)
    out["risk_level"] = pd.cut(probs, bins=[-0.001,0.30,0.70,1.001],
                               labels=["Low","Medium","High"]).astype(str)
    out["prediction_label"] = out["prediction"].map({0:"Legitimate",1:"Potential Fraud"})
    return out
