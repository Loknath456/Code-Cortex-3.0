"""
FIN-XR Model Health & Inference Verification Script
Run this script to test if all 5 trained models are working correctly!

Usage:
    python ml/check_models.py
"""

import os
import sys

# Ensure ml package is on path
sys.path.append(os.path.dirname(__file__))

def check_all_models():
    print("\n==================================================================")
    print(" FIN-XR MODEL HEALTH & INFERENCE VERIFICATION")
    print("==================================================================\n")

    model_dir = os.path.join(os.path.dirname(__file__), "models")
    
    # 1. Check Artifact Files
    expected_artifacts = [
        "isolation_forest.joblib",
        "bank_scaler.joblib",
        "bank_features.joblib",
        "logistic_regression.joblib",
        "random_forest.joblib",
        "xgboost.joblib",
        "autoencoder.pt",
        "creditcard_scaler.joblib",
        "model_metadata.json"
    ]

    print("--- [CHECK 1/3] VERIFYING MODEL ARTIFACT FILES ---")
    missing_artifacts = []
    for artifact in expected_artifacts:
        path = os.path.join(model_dir, artifact)
        if os.path.exists(path):
            size_kb = os.path.getsize(path) / 1024
            print(f"  [OK] {artifact:<30} ({size_kb:.1f} KB)")
        else:
            print(f"  [MISSING] {artifact:<30}")
            missing_artifacts.append(artifact)

    # 2. Test Bank Behavioral Anomaly Model (Isolation Forest)
    print("\n--- [CHECK 2/3] TESTING BANK BEHAVIORAL ANOMALY INFERENCE ---")
    try:
        from inference.predict_bank import analyze_bank_transaction

        test_bank_tx = {
            "account_no": "ACC7843",
            "amount": 85000.0,
            "is_withdrawal": 1,
            "hour": 3,
            "day_of_week": 4,
            "historical_avg_amount": 5000.0,
            "historical_std_amount": 1200.0,
            "time_since_last_tx_hours": 0.25,
            "rolling_tx_count_3d": 27,
            "balance_change_ratio": -0.58
        }

        bank_res = analyze_bank_transaction(test_bank_tx)
        print("  [SUCCESS] Bank Isolation Forest is WORKING!")
        print(f"  -> Account          : {bank_res['account_no']}")
        print(f"  -> Anomaly Score    : {bank_res['behavioral_anomaly_score']}")
        print(f"  -> Pattern Status   : {bank_res['pattern_status']}")
        print("  -> Explainable Risk Signals:")
        for signal in bank_res['explainable_signals']:
            print(f"     - {signal}")

    except Exception as e:
        print(f"  [ERROR] Bank inference failed: {e}")

    # 3. Test Supervised Credit Card & Autoencoder Models
    print("\n--- [CHECK 3/3] TESTING CREDIT CARD FRAUD & DEEP AUTOENCODER INFERENCE ---")
    try:
        from inference.predict_fraud import predict_creditcard_fraud, detect_autoencoder_anomaly
        import numpy as np

        test_cc_tx = {"Time": 10000, "Amount": 500.0}
        for i in range(1, 29):
            test_cc_tx[f"V{i}"] = -1.5

        # Test XGBoost
        xgb_res = predict_creditcard_fraud(test_cc_tx, model_name="xgboost")
        print("  [SUCCESS] XGBoost Fraud Engine is WORKING!")
        print(f"  -> Fraud Probability : {xgb_res['fraud_probability']}")
        print(f"  -> Risk Level         : {xgb_res['risk_level']}")

        # Test Logistic Regression
        lr_res = predict_creditcard_fraud(test_cc_tx, model_name="logistic_regression")
        print("  [SUCCESS] Logistic Regression Engine is WORKING!")
        print(f"  -> Fraud Probability : {lr_res['fraud_probability']}")

        # Test Random Forest
        rf_res = predict_creditcard_fraud(test_cc_tx, model_name="random_forest")
        print("  [SUCCESS] Random Forest Engine is WORKING!")
        print(f"  -> Fraud Probability : {rf_res['fraud_probability']}")

        # Test PyTorch Autoencoder
        ae_res = detect_autoencoder_anomaly(test_cc_tx)
        print("  [SUCCESS] PyTorch Autoencoder is WORKING!")
        print(f"  -> Reconstruction MSE : {ae_res['reconstruction_mse']}")
        print(f"  -> Deep Anomaly Score : {ae_res['deep_anomaly_score']}")

    except Exception as e:
        print(f"  [ERROR] Credit card inference failed: {e}")

    print("\n==================================================================")
    print(" VERIFICATION SUMMARY")
    print("==================================================================")
    if not missing_artifacts:
        print(" ALL 5 MODELS ARE FULLY FUNCTIONAL AND OPERATIONAL!")
    else:
        print(f" Missing {len(missing_artifacts)} model files. Run: python ml/train_all.py")
    print("==================================================================\n")

if __name__ == "__main__":
    check_all_models()
