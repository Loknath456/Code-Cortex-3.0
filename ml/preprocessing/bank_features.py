"""
FIN-XR Preprocessing & Behavioral Feature Engineering for bank.xlsx
Models: How does this account normally behave?
Adapted strictly to authoritative dataset schema.
"""

import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    'transaction_amount',
    'is_withdrawal',
    'transaction_hour',
    'day_of_week',
    'time_since_last_tx_hours',
    'rolling_tx_count_3d',
    'rolling_avg_amt_14d',
    'rolling_std_amt_14d',
    'amount_to_avg_ratio',
    'balance_change_ratio',
    'nocturnal_flag'
]

def load_and_preprocess_bank_data(filepath):
    """
    Loads, cleans, and parses bank.xlsx data.
    """
    print(f"[BANK PREPROCESS] Loading bank dataset from: {filepath}")
    df = pd.read_excel(filepath)
    initial_rows = len(df)

    # Standardize column names (strip whitespace)
    df.columns = [str(c).strip() for c in df.columns]

    # Map column aliases for dataset resilience
    column_mapping = {}
    for col in df.columns:
        c_upper = col.upper().replace('.', '').strip()
        if 'ACCOUNT' in c_upper:
            column_mapping[col] = 'Account No'
        elif c_upper in ['DATE', 'TRANSACTION DATE']:
            column_mapping[col] = 'Date'
        elif 'WITHDRAWAL' in c_upper:
            column_mapping[col] = 'WITHDRAWAL AMT'
        elif 'DEPOSIT' in c_upper:
            column_mapping[col] = 'DEPOSIT AMT'
        elif 'BALANCE' in c_upper:
            column_mapping[col] = 'BALANCE AMT'
        elif 'DETAILS' in c_upper:
            column_mapping[col] = 'Transaction Details'
        elif 'CHQ' in c_upper:
            column_mapping[col] = 'CHQ.NO'
        elif 'VALUE' in c_upper:
            column_mapping[col] = 'VALUE DATE'

    df = df.rename(columns=column_mapping)

    # Validate required columns
    req_cols = ['Account No', 'Date', 'WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']
    for req in req_cols:
        if req not in df.columns:
            raise KeyError(f"Missing required column in bank dataset: {req}. Found: {list(df.columns)}")

    # Parse Dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df = df.dropna(subset=['Date'])
    
    # Sort strictly by Account No and Date to prevent temporal leakage
    df = df.sort_values(by=['Account No', 'Date']).reset_index(drop=True)

    # Convert numeric columns safely
    for num_col in ['WITHDRAWAL AMT', 'DEPOSIT AMT', 'BALANCE AMT']:
        df[num_col] = pd.to_numeric(df[num_col], errors='coerce').fillna(0.0)

    # Remove duplicates
    df = df.drop_duplicates().reset_index(drop=True)
    
    final_rows = len(df)
    print(f"[BANK PREPROCESS] Processed {initial_rows} -> {final_rows} rows (Removed {initial_rows - final_rows} rows/duplicates).")
    return df

def create_bank_behavioral_features(df):
    """
    Engineers 11 behavioral features for Isolation Forest.
    """
    print("[BANK FEATURES] Engineering account behavioral pattern features...")
    df = df.copy()

    # 1. Transaction Amount
    df['transaction_amount'] = np.maximum(df['WITHDRAWAL AMT'], df['DEPOSIT AMT'])
    
    # 2. Deposit / Withdrawal Indicator (1 = Withdrawal, 0 = Deposit)
    df['is_withdrawal'] = (df['WITHDRAWAL AMT'] > 0).astype(int)

    # 3. Transaction Hour & 4. Day of Week
    df['transaction_hour'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek

    # Nocturnal Indicator (2 AM - 5 AM window)
    df['nocturnal_flag'] = df['transaction_hour'].apply(lambda h: 1 if 2 <= h <= 5 else 0)

    # Group by Account for temporal rolling features
    df['time_since_last_tx_hours'] = 0.0
    df['rolling_tx_count_3d'] = 1.0
    df['rolling_avg_amt_14d'] = df['transaction_amount']
    df['rolling_std_amt_14d'] = 0.0
    df['balance_change_ratio'] = 0.0

    processed_groups = []

    for acc_id, group in df.groupby('Account No', sort=False):
        group = group.copy()
        
        # Time Interval between consecutive transactions (hours)
        time_diff = group['Date'].diff().dt.total_seconds() / 3600.0
        group['time_since_last_tx_hours'] = time_diff.fillna(24.0)

        # Balance change ratio relative to previous balance
        prev_balance = group['BALANCE AMT'].shift(1).replace(0, np.nan)
        balance_diff = group['BALANCE AMT'] - prev_balance
        group['balance_change_ratio'] = (balance_diff / prev_balance).fillna(0.0).clip(-1.0, 5.0)

        # Temporal rolling statistics (using expanding window if rows < window)
        hist_amt = group['transaction_amount'].shift(1)
        
        group['rolling_avg_amt_14d'] = hist_amt.expanding(min_periods=1).mean().fillna(group['transaction_amount'])
        group['rolling_std_amt_14d'] = hist_amt.expanding(min_periods=1).std().fillna(0.0)
        group['rolling_tx_count_3d'] = group['transaction_amount'].expanding(min_periods=1).count()

        processed_groups.append(group)

    df_features = pd.concat(processed_groups, ignore_index=True)

    # Ratio of current transaction amount to account's historical average
    denom = df_features['rolling_avg_amt_14d'].replace(0, 1.0)
    df_features['amount_to_avg_ratio'] = (df_features['transaction_amount'] / denom).clip(0.0, 50.0)

    print(f"[BANK FEATURES] Extracted {len(FEATURE_COLUMNS)} behavioral features across {len(df_features)} transactions.")
    return df_features, FEATURE_COLUMNS
