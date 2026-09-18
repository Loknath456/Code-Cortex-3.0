"""
FIN-XR Synthetic Dataset Generator
Creates benchmark bank.xlsx and creditcard.csv datasets matching exact product schemas if raw files are absent.
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(DATA_DIR, exist_ok=True)

def generate_bank_dataset(num_records=15000, num_accounts=200):
    """
    Generates bank.xlsx with schema:
    - Account No
    - Date
    - Transaction Details
    - CHQ.NO
    - VALUE DATE
    - WITHDRAWAL AMT
    - DEPOSIT AMT
    - BALANCE AMT
    """
    print(f"[GENERATE] Creating bank.xlsx dataset (~{num_records} records)...")
    np.random.seed(42)

    accounts = [f"ACC{1000 + i}" for i in range(num_accounts)]
    details_pool = [
      "ATM Cash Withdrawal", "Direct Debit Salary", "ACH Merchant Deposit",
      "UPI Transfer", "NEFT Wire Out", "Online Shopping Payment",
      "Utility Bill Payment", "POS Card Swipe", "Interest Credit"
    ]

    records = []
    base_date = datetime(2026, 1, 1, 8, 0, 0)

    account_balances = {acc: np.random.uniform(50000, 200000) for acc in accounts}

    for i in range(num_records):
        acc = np.random.choice(accounts)
        days_offset = np.random.uniform(0, 180)
        
        # Inject off-hours nocturnal spikes for ~3% of records
        if np.random.rand() < 0.03:
            hour = np.random.choice([1, 2, 3, 4])
        else:
            hour = np.random.choice(range(8, 22))

        tx_time = base_date + timedelta(days=days_offset, hours=hour, minutes=np.random.randint(0, 60))
        val_time = tx_time + timedelta(hours=1)

        tx_detail = np.random.choice(details_pool)

        # 60% withdrawals, 40% deposits
        is_withdrawal = np.random.rand() < 0.6

        if is_withdrawal:
            # Inject high amount anomaly for 2% of withdrawals
            if np.random.rand() < 0.02:
                amount = np.random.uniform(70000, 150000)
            else:
                amount = np.random.exponential(scale=3000) + 200
            
            withdrawal_amt = round(amount, 2)
            deposit_amt = 0.0
            account_balances[acc] = max(1000, account_balances[acc] - withdrawal_amt)
        else:
            deposit_amt = round(np.random.exponential(scale=8000) + 1000, 2)
            withdrawal_amt = 0.0
            account_balances[acc] += deposit_amt

        chq_no = f"CHQ{np.random.randint(100000, 999999)}" if np.random.rand() < 0.3 else "-"

        records.append({
            "Account No": acc,
            "Date": tx_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Transaction Details": tx_detail,
            "CHQ.NO": chq_no,
            "VALUE DATE": val_time.strftime("%Y-%m-%d"),
            "WITHDRAWAL AMT": withdrawal_amt,
            "DEPOSIT AMT": deposit_amt,
            "BALANCE AMT": round(account_balances[acc], 2)
        })

    df_bank = pd.DataFrame(records)
    output_path = os.path.join(DATA_DIR, "bank.xlsx")
    df_bank.to_excel(output_path, index=False)
    print(f"[GENERATE] Saved bank.xlsx -> {output_path} (Shape: {df_bank.shape})")
    return output_path

def generate_creditcard_dataset(num_records=30000):
    """
    Generates creditcard.csv with schema:
    - Time
    - V1 to V28
    - Amount
    - Class (0 = legitimate, 1 = fraud ~0.17%)
    """
    print(f"[GENERATE] Creating creditcard.csv dataset (~{num_records} records)...")
    np.random.seed(42)

    # Time feature
    time_feature = np.sort(np.random.uniform(0, 172800, num_records))

    # PCA Features V1 to V28
    v_features = np.random.normal(0, 1, size=(num_records, 28))

    # Amount
    amount = np.random.exponential(scale=88, size=num_records)

    # Class target (0.17% fraud rate)
    num_fraud = int(num_records * 0.00172)
    fraud_indices = np.random.choice(num_records, size=num_fraud, replace=False)

    class_target = np.zeros(num_records, dtype=int)
    class_target[fraud_indices] = 1

    # Induce shift in fraud records for synthetic PCA features
    v_features[fraud_indices, 0:5] += np.random.normal(-3, 1, size=(num_fraud, 5))
    v_features[fraud_indices, 10:14] += np.random.normal(3, 1, size=(num_fraud, 4))
    amount[fraud_indices] = np.random.uniform(150, 2000, size=num_fraud)

    data_dict = {"Time": time_feature}
    for i in range(1, 29):
        data_dict[f"V{i}"] = v_features[:, i-1]

    data_dict["Amount"] = amount
    data_dict["Class"] = class_target

    df_cc = pd.DataFrame(data_dict)
    output_path = os.path.join(DATA_DIR, "creditcard.csv")
    df_cc.to_csv(output_path, index=False)
    print(f"[GENERATE] Saved creditcard.csv -> {output_path} (Shape: {df_cc.shape}, Fraud: {num_fraud})")
    return output_path

if __name__ == "__main__":
    generate_bank_dataset()
    generate_creditcard_dataset()
