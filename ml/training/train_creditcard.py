"""
FIN-XR SUPERVISED FRAUD MODELS (creditcard.csv)
Trains Logistic Regression (Baseline), Random Forest, and XGBoost (Primary).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.creditcard_features import load_and_preprocess_creditcard_data, prepare_creditcard_splits
from evaluation.metrics import compute_fraud_metrics, print_metrics_summary

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

def train_supervised_fraud_models(data_path=None):
    """
    Trains Logistic Regression, Random Forest, and XGBoost models.
    """
    if data_path is None:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv"))

    print("\n==================================================")
    print(" TRAINING SUPERVISED FRAUD MODELS (CREDITCARD)")
    print("==================================================")

    # 1. Load & Split Data
    df_cc = load_and_preprocess_creditcard_data(data_path)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = prepare_creditcard_splits(df_cc)

    # Save Scaler & Features
    joblib.dump(scaler, os.path.join(MODEL_DIR, "creditcard_scaler.joblib"))
    joblib.dump(feature_names, os.path.join(MODEL_DIR, "creditcard_features.joblib"))

    results = {}

    # =========================================================================
    # MAIN MODEL 2: LOGISTIC REGRESSION (BASELINE)
    # =========================================================================
    print("\n[TRAIN] Training Main Model 2: Logistic Regression (class_weight='balanced')...")
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)

    lr_val_probs = lr.predict_proba(X_val)[:, 1]
    lr_test_probs = lr.predict_proba(X_test)[:, 1]

    lr_metrics_val = compute_fraud_metrics(y_val, lr_val_probs)
    lr_metrics_test = compute_fraud_metrics(y_test, lr_test_probs)
    print_metrics_summary("Logistic Regression (Test Set)", lr_metrics_test)

    joblib.dump(lr, os.path.join(MODEL_DIR, "logistic_regression.joblib"))
    results["logistic_regression"] = {
        "val": lr_metrics_val, "test": lr_metrics_test, "test_probs": lr_test_probs
    }

    # =========================================================================
    # MAIN MODEL 3: RANDOM FOREST
    # =========================================================================
    print("\n[TRAIN] Training Main Model 3: Random Forest (class_weight='balanced_subsample')...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    rf_val_probs = rf.predict_proba(X_val)[:, 1]
    rf_test_probs = rf.predict_proba(X_test)[:, 1]

    rf_metrics_val = compute_fraud_metrics(y_val, rf_val_probs)
    rf_metrics_test = compute_fraud_metrics(y_test, rf_test_probs)
    print_metrics_summary("Random Forest (Test Set)", rf_metrics_test)

    joblib.dump(rf, os.path.join(MODEL_DIR, "random_forest.joblib"))

    # Save RF feature importance
    rf_importances = pd.Series(rf.feature_importances_, index=feature_names).sort_values(ascending=False)
    rf_importances.to_csv(os.path.join(OUTPUT_DIR, "rf_feature_importance.csv"))

    results["random_forest"] = {
        "val": rf_metrics_val, "test": rf_metrics_test, "test_probs": rf_test_probs
    }

    # =========================================================================
    # MAIN MODEL 4: XGBOOST (PRIMARY MODEL)
    # =========================================================================
    print("\n[TRAIN] Training Main Model 4: XGBoost (Primary Fraud Engine)...")
    
    # Calculate scale_pos_weight strictly from train split
    num_neg = (y_train == 0).sum()
    num_pos = (y_train == 1).sum()
    scale_pos = num_neg / max(1, num_pos)

    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.08,
        scale_pos_weight=scale_pos,
        subsample=0.85,
        colsample_bytree=0.85,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1
    )

    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    xgb_val_probs = xgb_model.predict_proba(X_val)[:, 1]
    xgb_test_probs = xgb_model.predict_proba(X_test)[:, 1]

    xgb_metrics_val = compute_fraud_metrics(y_val, xgb_val_probs)
    xgb_metrics_test = compute_fraud_metrics(y_test, xgb_test_probs)
    print_metrics_summary("XGBoost (Test Set)", xgb_metrics_test)

    joblib.dump(xgb_model, os.path.join(MODEL_DIR, "xgboost.joblib"))

    # Save XGBoost feature importance
    xgb_importances = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    xgb_importances.to_csv(os.path.join(OUTPUT_DIR, "xgb_feature_importance.csv"))

    results["xgboost"] = {
        "val": xgb_metrics_val, "test": xgb_metrics_test, "test_probs": xgb_test_probs
    }

    # Package Ground Truth & Predictions for Evaluation Pipeline
    y_test_path = os.path.join(OUTPUT_DIR, "y_test.npy")
    np.save(y_test_path, y_test)

    return results, (X_test, y_test)

if __name__ == "__main__":
    train_supervised_fraud_models()
