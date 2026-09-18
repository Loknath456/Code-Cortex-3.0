"""
FIN-XR MAIN MODEL 1: Isolation Forest Training on bank.xlsx
Purpose: Transaction Behavioral Anomaly Detection
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.bank_features import load_and_preprocess_bank_data, create_bank_behavioral_features

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "predictions"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def map_score_to_pattern_status(score):
    """
    Maps normalized anomaly score (0.00 to 1.00) to Product UI pattern status.
    Note: These are product UI threshold categories, not calibrated probabilities.
    """
    if score < 0.40:
        return "Normal"
    elif score < 0.60:
        return "Minor Deviation"
    elif score < 0.80:
        return "Behavior Shift"
    else:
        return "High Deviation"

def train_bank_isolation_forest(data_path=None):
    """
    Trains Isolation Forest on bank.xlsx dataset.
    """
    if data_path is None:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "bank.xlsx"))

    print("\n==================================================")
    print(" TRAINING MAIN MODEL 1: ISOLATION FOREST (BANK)")
    print("==================================================")

    # 1. Preprocess & Feature Engineering
    df_raw = load_and_preprocess_bank_data(data_path)
    df_features, feature_cols = create_bank_behavioral_features(df_raw)

    X = df_features[feature_cols].copy()

    # Handle any infs or NaNs cleanly
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    # 2. Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Train Isolation Forest
    print("[TRAIN BANK] Fitting Isolation Forest (n_estimators=300, contamination=0.05)...")
    model = IsolationForest(
        n_estimators=300,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_scaled)

    # 4. Generate Scores
    # decision_function gives negative score for anomalies, positive for normal
    raw_scores = -model.decision_function(X_scaled)

    # Min-Max normalize raw score into [0.0, 1.0] interval
    min_s, max_s = raw_scores.min(), raw_scores.max()
    norm_scores = (raw_scores - min_s) / (max_s - min_s + 1e-8)

    df_features["raw_anomaly_score"] = raw_scores
    df_features["normalized_anomaly_score"] = norm_scores
    df_features["pattern_status"] = [map_score_to_pattern_status(s) for s in norm_scores]

    # 5. Summary Statistics
    status_counts = df_features["pattern_status"].value_counts().to_dict()
    print("[TRAIN BANK] Anomaly Classification Summary:")
    for status, count in status_counts.items():
        print(f"  - {status}: {count} ({count/len(df_features):.2%})")

    # 6. Save Artifacts
    model_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
    scaler_path = os.path.join(MODEL_DIR, "bank_scaler.joblib")
    features_path = os.path.join(MODEL_DIR, "bank_features.joblib")
    scored_output_path = os.path.join(OUTPUT_DIR, "bank_scored_transactions.csv")

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)
    df_features.to_csv(scored_output_path, index=False)

    print(f"[SAVE] Saved Isolation Forest model -> {model_path}")
    print(f"[SAVE] Saved Bank Scaler -> {scaler_path}")
    print(f"[SAVE] Saved Scored Data -> {scored_output_path}")

    return {
        "model": model,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "status_counts": status_counts,
        "num_records": len(df_features)
    }

if __name__ == "__main__":
    train_bank_isolation_forest()
