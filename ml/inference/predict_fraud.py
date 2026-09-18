"""
FIN-XR Inference Engine: Credit Card Fraud & Deep Learning Anomaly Detection
Provides predict_creditcard_fraud and detect_autoencoder_anomaly.
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))

_xgb_model = None
_rf_model = None
_lr_model = None
_cc_scaler = None
_cc_features = None
_ae_checkpoint = None

def _load_creditcard_artifacts():
    global _xgb_model, _rf_model, _lr_model, _cc_scaler, _cc_features
    if _xgb_model is None:
        _xgb_model = joblib.load(os.path.join(MODEL_DIR, "xgboost.joblib"))
        _rf_model = joblib.load(os.path.join(MODEL_DIR, "random_forest.joblib"))
        _lr_model = joblib.load(os.path.join(MODEL_DIR, "logistic_regression.joblib"))
        _cc_scaler = joblib.load(os.path.join(MODEL_DIR, "creditcard_scaler.joblib"))
        _cc_features = joblib.load(os.path.join(MODEL_DIR, "creditcard_features.joblib"))

def predict_creditcard_fraud(tx_dict, model_name="xgboost"):
    """
    Performs supervised fraud prediction on a credit card transaction dictionary.
    """
    _load_creditcard_artifacts()

    # Construct DataFrame in exact feature order
    df_single = pd.DataFrame([tx_dict])[_cc_features].copy()

    # Scale Time & Amount
    cols_to_scale = ['Time', 'Amount']
    df_single[cols_to_scale] = _cc_scaler.transform(df_single[cols_to_scale])

    # Select model
    if model_name == "xgboost":
        model = _xgb_model
    elif model_name == "random_forest":
        model = _rf_model
    else:
        model = _lr_model

    prob = float(model.predict_proba(df_single)[0, 1])

    if prob >= 0.75:
        risk_level = "HIGH"
    elif prob >= 0.40:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return {
        "fraud_probability": round(prob, 4),
        "risk_level": risk_level,
        "model_used": model_name
    }

def detect_autoencoder_anomaly(tx_dict):
    """
    Runs PyTorch Autoencoder reconstruction anomaly detection.
    """
    _load_creditcard_artifacts()
    
    ae_path = os.path.join(MODEL_DIR, "autoencoder.pt")
    if not os.path.exists(ae_path):
        raise FileNotFoundError("Autoencoder model not found. Run train_autoencoder.py first.")

    checkpoint = torch.load(ae_path, map_location="cpu")
    input_dim = checkpoint["input_dim"]
    anomaly_thresh = checkpoint["anomaly_threshold"]

    # Reconstruct architecture
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from training.train_autoencoder import TransactionAutoencoder

    ae_model = TransactionAutoencoder(input_dim)
    ae_model.load_state_dict(checkpoint["state_dict"])
    ae_model.eval()

    # Format input tensor
    df_single = pd.DataFrame([tx_dict])[_cc_features].copy()
    cols_to_scale = ['Time', 'Amount']
    df_single[cols_to_scale] = _cc_scaler.transform(df_single[cols_to_scale])

    x_tensor = torch.from_numpy(df_single.values.astype(np.float32))

    with torch.no_grad():
        reconstructed, _ = ae_model(x_tensor)
        mse_loss = float(torch.mean((reconstructed - x_tensor) ** 2).item())

    # Deep anomaly score
    deep_score = float(np.clip(mse_loss / (anomaly_thresh * 1.5 + 1e-8), 0.0, 1.0))

    return {
        "reconstruction_mse": round(mse_loss, 6),
        "reconstruction_threshold": round(anomaly_thresh, 6),
        "deep_anomaly_score": round(deep_score, 4),
        "is_reconstruction_anomaly": mse_loss > anomaly_thresh
    }

if __name__ == "__main__":
    sample_tx = {"Time": 10000, "Amount": 500.0}
    for i in range(1, 29):
        sample_tx[f"V{i}"] = np.random.normal(-2, 1)

    print("\n[INFERENCE TEST] Supervised Fraud Result:")
    print(predict_creditcard_fraud(sample_tx, "xgboost"))

    print("\n[INFERENCE TEST] Autoencoder Deep Anomaly Result:")
    print(detect_autoencoder_anomaly(sample_tx))
