"""
FIN-XR Preprocessing & Stratified Splitting for creditcard.csv
Strictly avoids data leakage: Scaler is fitted ONLY on train data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_and_preprocess_creditcard_data(filepath):
    """
    Loads creditcard.csv and checks schema.
    """
    print(f"[CREDITCARD PREPROCESS] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    print(f"[CREDITCARD PREPROCESS] Raw Shape: {df.shape}")

    if 'Class' not in df.columns:
        raise KeyError("Missing target column 'Class' in creditcard dataset.")

    # Missing value check
    missing_count = df.isnull().sum().sum()
    print(f"[CREDITCARD PREPROCESS] Missing values: {missing_count}")
    
    # Target distribution
    fraud_count = df['Class'].sum()
    total_count = len(df)
    print(f"[CREDITCARD PREPROCESS] Class balance: Legitimate={total_count - fraud_count}, Fraud={fraud_count} ({fraud_count/total_count:.4%})")

    return df

def prepare_creditcard_splits(df, random_state=42):
    """
    Stratified 70% Train / 15% Validation / 15% Test split.
    Fits StandardScaler ONLY on training set.
    """
    X = df.drop(columns=['Class']).copy()
    y = df['Class'].values

    feature_names = list(X.columns)

    # First split: 70% Train, 30% Temp (Val + Test)
    X_train_raw, X_temp_raw, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=random_state, stratify=y
    )

    # Second split: Split 30% Temp into 15% Validation, 15% Test
    X_val_raw, X_test_raw, y_val, y_test = train_test_split(
        X_temp_raw, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
    )

    print(f"[CREDITCARD SPLIT] Train: {X_train_raw.shape[0]} | Val: {X_val_raw.shape[0]} | Test: {X_test_raw.shape[0]}")

    # Scale Time & Amount ONLY on train set to prevent data leakage
    scaler = StandardScaler()
    
    # Scale Amount and Time (columns 0 and -1)
    cols_to_scale = ['Time', 'Amount']
    
    X_train = X_train_raw.copy()
    X_val = X_val_raw.copy()
    X_test = X_test_raw.copy()

    scaler.fit(X_train[cols_to_scale])

    X_train[cols_to_scale] = scaler.transform(X_train[cols_to_scale])
    X_val[cols_to_scale] = scaler.transform(X_val[cols_to_scale])
    X_test[cols_to_scale] = scaler.transform(X_test[cols_to_scale])

    return (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names
