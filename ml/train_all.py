"""
FIN-XR MASTER TRAINING PIPELINE RUNNER
Executes dataset verification, preprocessing, model training, evaluation, plotting, and metadata generation.
"""

import os
import sys
import time

sys.path.append(os.path.dirname(__file__))

from generate_datasets import generate_bank_dataset, generate_creditcard_dataset
from training.train_bank import train_bank_isolation_forest
from training.train_creditcard import train_supervised_fraud_models
from training.train_autoencoder import train_autoencoder
from evaluation.evaluate_models import generate_all_evaluation_plots, create_model_metadata_json
from inference.predict_bank import analyze_bank_transaction
from inference.predict_fraud import predict_creditcard_fraud, detect_autoencoder_anomaly

def run_master_training_pipeline():
    start_time = time.time()
    print("\n==================================================================")
    print(" FIN-XR MACHINE LEARNING LAYER — MASTER PIPELINE EXECUTION")
    print(" AI-Powered Transaction Pattern Intelligence & Risk Analytics")
    print("==================================================================\n")

    data_dir = os.path.join(os.path.dirname(__file__), "data")
    bank_path = os.path.join(data_dir, "bank.xlsx")
    creditcard_path = os.path.join(data_dir, "creditcard.csv")

    # Step 1: Dataset Verification & Auto-bootstrap if missing
    print("--- [STEP 1/6] DATASET VERIFICATION ---")
    if not os.path.exists(bank_path):
        print(f"[VERIFY] bank.xlsx not found at {bank_path}. Bootstrapping dataset...")
        generate_bank_dataset()
    else:
        print(f"[VERIFY] Found existing bank.xlsx -> {bank_path}")

    if not os.path.exists(creditcard_path):
        print(f"[VERIFY] creditcard.csv not found at {creditcard_path}. Bootstrapping dataset...")
        generate_creditcard_dataset()
    else:
        print(f"[VERIFY] Found existing creditcard.csv -> {creditcard_path}")

    # Step 2: Main Model 1 — Isolation Forest
    print("\n--- [STEP 2/6] TRAINING MAIN MODEL 1: ISOLATION FOREST ---")
    bank_results = train_bank_isolation_forest(bank_path)

    # Step 3: Main Models 2, 3, 4 — Logistic Regression, Random Forest, XGBoost
    print("\n--- [STEP 3/6] TRAINING SUPERVISED FRAUD MODELS (LR, RF, XGB) ---")
    supervised_results, (X_test, y_test) = train_supervised_fraud_models(creditcard_path)

    # Step 4: Deep Learning Component — PyTorch Autoencoder
    print("\n--- [STEP 4/6] TRAINING DEEP LEARNING COMPONENT (PYTORCH AUTOENCODER) ---")
    ae_results = train_autoencoder(creditcard_path, epochs=10)

    # Step 5: Visualizations & Metadata Contract
    print("\n--- [STEP 5/6] GENERATING VISUALIZATIONS & METADATA CONTRACT ---")
    
    # Package prediction probabilities for comparison
    predictions_dict = {
        "logistic_regression": supervised_results["logistic_regression"]["test_probs"],
        "random_forest": supervised_results["random_forest"]["test_probs"],
        "xgboost": supervised_results["xgboost"]["test_probs"],
        "autoencoder": ae_results["test_deep_scores"]
    }

    import pandas as pd
    scored_bank_path = os.path.join(os.path.dirname(__file__), "outputs", "predictions", "bank_scored_transactions.csv")
    bank_df = pd.read_csv(scored_bank_path) if os.path.exists(scored_bank_path) else None

    generate_all_evaluation_plots(y_test, predictions_dict, bank_df)

    all_metrics = {
        "isolation_forest_status_counts": bank_results["status_counts"],
        "logistic_regression": supervised_results["logistic_regression"]["test"],
        "random_forest": supervised_results["random_forest"]["test"],
        "xgboost": supervised_results["xgboost"]["test"],
        "autoencoder": ae_results["test_metrics"]
    }
    create_model_metadata_json(all_metrics)

    # Step 6: Inference Smoke Tests
    print("\n--- [STEP 6/6] INFERENCE LAYER SMOKE TESTS ---")
    
    sample_bank_tx = {
        "account_no": "ACC7843",
        "amount": 85000.0,
        "is_withdrawal": 1,
        "hour": 3,
        "day_of_week": 4,
        "historical_avg_amount": 5000.0,
        "time_since_last_tx_hours": 0.25,
        "rolling_tx_count_3d": 27,
        "balance_change_ratio": -0.58
    }
    bank_inference_out = analyze_bank_transaction(sample_bank_tx)
    print(f" [SMOKE TEST] Bank Anomaly Engine Result: {bank_inference_out['pattern_status']} (Score: {bank_inference_out['behavioral_anomaly_score']})")

    sample_cc_tx = {"Time": 10000, "Amount": 500.0}
    for i in range(1, 29): sample_cc_tx[f"V{i}"] = -1.5
    
    cc_inference_out = predict_creditcard_fraud(sample_cc_tx, "xgboost")
    print(f" [SMOKE TEST] XGBoost Fraud Engine Result: Risk Level {cc_inference_out['risk_level']} (Prob: {cc_inference_out['fraud_probability']})")

    ae_inference_out = detect_autoencoder_anomaly(sample_cc_tx)
    print(f" [SMOKE TEST] PyTorch Autoencoder Deep Anomaly Score: {ae_inference_out['deep_anomaly_score']}")

    elapsed = time.time() - start_time
    print("\n==================================================================")
    print(f" SUCCESS: FIN-XR MASTER ML PIPELINE COMPLETED IN {elapsed:.2f} SECONDS")
    print("==================================================================\n")

if __name__ == "__main__":
    run_master_training_pipeline()
