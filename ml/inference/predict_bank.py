"""
FIN-XR Inference Engine: Bank Transaction Behavioral Pattern Intelligence
Generates human-readable behavioral explanations from model signals.
"""

import os
import joblib
import numpy as np
import pandas as pd

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

_model = None
_scaler = None
_feature_cols = None

def _load_bank_artifacts():
    global _model, _scaler, _feature_cols
    if _model is None:
        model_path = os.path.join(MODEL_DIR, "isolation_forest.joblib")
        scaler_path = os.path.join(MODEL_DIR, "bank_scaler.joblib")
        feat_path = os.path.join(MODEL_DIR, "bank_features.joblib")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}. Please run train_bank.py first.")

        _model = joblib.load(model_path)
        _scaler = joblib.load(scaler_path)
        _feature_cols = joblib.load(feat_path)

def analyze_bank_transaction(tx):
    """
    Analyzes a bank transaction dictionary for behavioral anomalies.
    Input example:
    {
        "account_no": "ACC7843",
        "amount": 85000.0,
        "is_withdrawal": 1,
        "hour": 3,
        "day_of_week": 4,
        "historical_avg_amount": 4500.0,
        "historical_std_amount": 1200.0,
        "time_since_last_tx_hours": 0.25,
        "rolling_tx_count_3d": 27,
        "balance_change_ratio": -0.58
    }
    """
    _load_bank_artifacts()

    # Feature preparation
    hist_avg = tx.get("historical_avg_amount", 5000.0)
    amt = tx.get("amount", 0.0)
    amt_ratio = amt / (hist_avg if hist_avg > 0 else 1.0)
    hour = tx.get("hour", 12)

    feat_dict = {
        "transaction_amount": amt,
        "is_withdrawal": tx.get("is_withdrawal", 1),
        "transaction_hour": hour,
        "day_of_week": tx.get("day_of_week", 1),
        "time_since_last_tx_hours": tx.get("time_since_last_tx_hours", 12.0),
        "rolling_tx_count_3d": tx.get("rolling_tx_count_3d", 3.0),
        "rolling_avg_amt_14d": hist_avg,
        "rolling_std_amt_14d": tx.get("historical_std_amount", 1000.0),
        "amount_to_avg_ratio": min(50.0, amt_ratio),
        "balance_change_ratio": tx.get("balance_change_ratio", 0.0),
        "nocturnal_flag": 1 if 2 <= hour <= 5 else 0
    }

    df_single = pd.DataFrame([feat_dict])[_feature_cols]
    X_scaled = _scaler.transform(df_single)

    raw_score = float(-_model.decision_function(X_scaled)[0])
    
    # Normalize score (approximation based on training range)
    norm_score = float(np.clip((raw_score + 0.2) / 0.5, 0.0, 1.0))

    # Pattern Status Mapping
    if norm_score < 0.40:
        pattern_status = "Normal"
    elif norm_score < 0.60:
        pattern_status = "Minor Deviation"
    elif norm_score < 0.80:
        pattern_status = "Behavior Shift"
    else:
        pattern_status = "High Deviation"

    # Human-Readable Explainable Risk Signals
    signals = []
    if amt_ratio > 3.0:
        signals.append(f"Transaction amount ({amt:,.2f}) is {amt_ratio:.1f}x above the account's historical average.")
    if 2 <= hour <= 5:
        signals.append("Transaction occurred outside the account's typical active operating hours (2 AM - 5 AM nocturnal window).")
    if tx.get("rolling_tx_count_3d", 0) > 15:
        signals.append(f"Transaction frequency is unusually high ({tx.get('rolling_tx_count_3d')} transactions within 3-day window).")
    if tx.get("time_since_last_tx_hours", 24) < 0.5:
        signals.append(f"Transaction interval ({tx.get('time_since_last_tx_hours')*60:.0f} mins) is substantially shorter than the account's usual pattern.")
    if abs(tx.get("balance_change_ratio", 0.0)) > 0.4:
        signals.append(f"Balance changed significantly ({tx.get('balance_change_ratio')*100:.1f}%) compared with historical behavior.")

    if not signals:
        signals.append("Transaction aligns with established account historical baseline.")

    return {
        "account_no": tx.get("account_no", "UNKNOWN"),
        "raw_anomaly_score": round(raw_score, 4),
        "behavioral_anomaly_score": round(norm_score, 4),
        "pattern_status": pattern_status,
        "explainable_signals": signals
    }

if __name__ == "__main__":
    test_tx = {
        "account_no": "ACC7843",
        "amount": 85000.0,
        "is_withdrawal": 1,
        "hour": 3,
        "day_of_week": 4,
        "historical_avg_amount": 5000.0,
        "time_since_last_tx_hours": 0.2,
        "rolling_tx_count_3d": 27,
        "balance_change_ratio": -0.58
    }
    result = analyze_bank_transaction(test_tx)
    print("\n[INFERENCE TEST] Bank Transaction Analysis Output:")
    print(result)
