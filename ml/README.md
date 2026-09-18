# FIN-XR — Machine Learning System & Risk Analytics Engine

> **Tagline**: *"Learn the Pattern. Detect the Change. Explain the Risk."*  
> **Core Principle**: 80% Transaction Intelligence, 20% Fraud/Risk Intelligence.

---

## 1. Project Overview & Architecture

FIN-XR is an AI-Powered Transaction Pattern Intelligence & Risk Analytics platform. It learns historical transaction behavior, establishes behavioral baselines, identifies subtle deviations from normal behavior, and provides explainable risk intelligence.

### The Two Datasets & ML Tasks

The system operates on **two distinct datasets** representing fundamentally different ML problems:

1. **`bank.xlsx` (Transaction Intelligence Dataset)**:
   - Contains historical bank account transactions (Account No, Date, Transaction Details, Withdrawal, Deposit, Balance).
   - **No Supervised Fraud Target**: Unsupervised behavioral anomaly detection using **Isolation Forest**.
   - **Goal**: Model "How does this account normally behave?" and output normalized anomaly scores mapped to product UI statuses: `Normal`, `Minor Deviation`, `Behavior Shift`, `High Deviation`.

2. **`creditcard.csv` (Supervised Fraud Benchmark Dataset)**:
   - Contains credit card transactions with PCA features (`V1`..`V28`), `Time`, `Amount`, and supervised target `Class` (0 = Legitimate, 1 = Fraud ~0.17%).
   - **Supervised & Deep Learning Models**:
     - **Logistic Regression**: Baseline linear model (`class_weight="balanced"`).
     - **Random Forest**: Non-linear tree ensemble (`class_weight="balanced_subsample"`).
     - **XGBoost**: Primary supervised fraud engine (`scale_pos_weight`, `aucpr` metric).
     - **PyTorch Autoencoder**: Deep learning anomaly detection (Trained ONLY on legitimate `Class == 0` transactions).

### Why the Datasets Are NOT Merged
`bank.xlsx` tracks multi-period account-level banking activity without ground-truth fraud labels, whereas `creditcard.csv` tracks single-transaction credit card authorizations with binary fraud labels. Merging them row-by-row would introduce data corruption and artificial leakage. They are processed separately and exposed as complementary intelligence signals.

---

## 2. Directory Structure

```
FIN-XR/
├── ml/
│   ├── data/
│   │   ├── bank.xlsx
│   │   └── creditcard.csv
│   ├── models/
│   │   ├── isolation_forest.joblib
│   │   ├── bank_scaler.joblib
│   │   ├── bank_features.joblib
│   │   ├── logistic_regression.joblib
│   │   ├── random_forest.joblib
│   │   ├── xgboost.joblib
│   │   ├── autoencoder.pt
│   │   ├── autoencoder_scaler.joblib
│   │   ├── creditcard_scaler.joblib
│   │   └── model_metadata.json
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── bank_features.py
│   │   └── creditcard_features.py
│   ├── training/
│   │   ├── __init__.py
│   │   ├── train_bank.py
│   │   ├── train_creditcard.py
│   │   └── train_autoencoder.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── evaluate_models.py
│   │   └── metrics.py
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predict_bank.py
│   │   └── predict_fraud.py
│   ├── notebooks/
│   │   └── EDA.ipynb
│   ├── outputs/
│   │   ├── metrics/
│   │   ├── plots/
│   │   └── predictions/
│   ├── generate_datasets.py
│   ├── train_all.py
│   ├── requirements.txt
│   └── README.md
└── backend/
    └── [FastAPI Integration Layer]
```

---

## 3. Behavioral Feature Engineering (`bank_features.py`)

To prevent temporal leakage, features are constructed using expanding/rolling historical windows (`shift(1)`):

1. **`transaction_amount`**: Gross withdrawal or deposit value.
2. **`is_withdrawal`**: Binary indicator (1 = Withdrawal/Payout, 0 = Deposit).
3. **`transaction_hour`**: Hour of transaction (0–23).
4. **`day_of_week`**: Day index (0 = Mon, 6 = Sun).
5. **`nocturnal_flag`**: Off-hours operating window indicator (1 if 2 AM–5 AM).
6. **`time_since_last_tx_hours`**: Time interval between consecutive account transactions.
7. **`rolling_tx_count_3d`**: Transaction burst velocity (3-day rolling count).
8. **`rolling_avg_amt_14d`**: Account's 14-day historical baseline transaction amount.
9. **`rolling_std_amt_14d`**: Account's historical amount volatility.
10. **`amount_to_avg_ratio`**: Current transaction amount relative to account's historical average.
11. **`balance_change_ratio`**: Percentage shift in account balance relative to prior balance.

---

## 4. Why PR-AUC Matters for Imbalanced Fraud Data

In financial fraud detection, legitimate transactions outnumber fraudulent ones (~99.83% vs 0.17%). Accuracy is misleading (a naive classifier predicting all 0s gets 99.83% accuracy).

**PR-AUC (Precision-Recall Area Under Curve)** evaluates Precision ($TP / (TP + FP)$) against Recall ($TP / (TP + FN)$) without being inflated by true negatives ($TN$). It is our primary optimization metric for supervised models.

---

## 5. PyTorch Autoencoder Deep Anomaly Detection

The PyTorch Autoencoder uses representation learning:
- **Training Strategy**: Trained **strictly on legitimate transactions** (`Class == 0`).
- **Architecture**: `Input -> 64 -> 32 -> 16 (Latent) -> 32 -> 64 -> Input`.
- **Loss Function**: Mean Squared Error ($MSE(x, \hat{x})$).
- **Anomaly Logic**: Legitimate transactions reconstruct with minimal loss. Anomalies/fraudulent patterns deviate from learned representations, yielding high reconstruction errors.

---

## 6. How to Run Training & Reproduce Results

### Install Dependencies
```bash
pip install -r ml/requirements.txt
```

### Run Full Pipeline
To execute dataset verification, preprocessing, model training, evaluation, plotting, and metadata generation in a single command:

```bash
python ml/train_all.py
```

### Run Individual Modules
```bash
# Generate synthetic datasets if needed
python ml/generate_datasets.py

# Train Isolation Forest on bank dataset
python ml/training/train_bank.py

# Train Supervised Models (LR, RF, XGBoost) on credit card dataset
python ml/training/train_creditcard.py

# Train PyTorch Autoencoder
python ml/training/train_autoencoder.py
```

---

## 7. Inference Layer & FastAPI Integration Contract

The inference module provides clean functions for downstream FastAPI endpoints:

### Bank Transaction Analysis:
```python
from ml.inference.predict_bank import analyze_bank_transaction

tx = {
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

result = analyze_bank_transaction(tx)
# Output:
# {
#     "account_no": "ACC7843",
#     "raw_anomaly_score": 0.3842,
#     "behavioral_anomaly_score": 0.87,
#     "pattern_status": "High Deviation",
#     "explainable_signals": [
#         "Transaction amount (85,000.00) is 17.0x above historical average.",
#         "Transaction occurred outside active hours (2 AM - 5 AM nocturnal window).",
#         "Transaction frequency is unusually high (27 transactions in 3 days)."
#     ]
# }
```

### Credit Card Fraud Prediction:
```python
from ml.inference.predict_fraud import predict_creditcard_fraud, detect_autoencoder_anomaly

result_xgb = predict_creditcard_fraud(sample_tx, model_name="xgboost")
result_ae = detect_autoencoder_anomaly(sample_tx)
```
