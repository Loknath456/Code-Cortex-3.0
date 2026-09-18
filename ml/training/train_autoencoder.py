"""
FIN-XR DEEP LEARNING COMPONENT: PyTorch Autoencoder
Dataset: creditcard.csv (Trained ONLY on Legitimate Transactions Class == 0)
Purpose: Deep-Learning Anomaly Detection & Representation Learning
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preprocessing.creditcard_features import load_and_preprocess_creditcard_data, prepare_creditcard_splits
from evaluation.metrics import compute_fraud_metrics, print_metrics_summary

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------------------------
# PYTORCH AUTOENCODER ARCHITECTURE
# -----------------------------------------------------------------------------
class TransactionAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(TransactionAutoencoder, self).__init__()
        
        # Encoder: Input -> 64 -> 32 -> 16
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU()
        )
        
        # Decoder: 16 -> 32 -> 64 -> Input
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed, latent

def train_autoencoder(data_path=None, epochs=15, batch_size=256, lr=1e-3):
    """
    Trains PyTorch Autoencoder on legitimate creditcard transactions.
    """
    if data_path is None:
        data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv"))

    print("\n==================================================")
    print(" TRAINING DEEP LEARNING COMPONENT: AUTOENCODER")
    print("==================================================")

    # Set Reproducibility Seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # 1. Load Data
    df_cc = load_and_preprocess_creditcard_data(data_path)
    (X_train, y_train), (X_val, y_val), (X_test, y_test), scaler, feature_names = prepare_creditcard_splits(df_cc)

    # CRITICAL: Filter ONLY Legitimate Transactions (Class == 0) for Training
    X_train_legit = X_train[y_train == 0].values.astype(np.float32)
    X_val_all = X_val.values.astype(np.float32)
    X_test_all = X_test.values.astype(np.float32)

    input_dim = X_train_legit.shape[1]

    # DataLoader
    train_dataset = TensorDataset(torch.from_numpy(X_train_legit))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Instantiate Model, Loss, Optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[AUTOENCODER] Using PyTorch Device: {device}")

    model = TransactionAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    # 2. Training Loop
    model.train()
    print(f"[AUTOENCODER] Training PyTorch Autoencoder on {len(X_train_legit)} legitimate records for {epochs} epochs...")

    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        for batch in train_loader:
            x_batch = batch[0].to(device)
            
            optimizer.zero_grad()
            reconstructed, _ = model(x_batch)
            loss = criterion(reconstructed, x_batch)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * x_batch.size(0)

        epoch_loss = running_loss / len(X_train_legit)
        if epoch % 3 == 0 or epoch == epochs:
            print(f"  Epoch [{epoch:02d}/{epochs:02d}] - Reconstruction MSE Loss: {epoch_loss:.6f}")

    # 3. Calculate Reconstruction Error Threshold on Legitimate Validation Set
    model.eval()
    with torch.no_grad():
        x_legit_tensor = torch.from_numpy(X_train_legit).to(device)
        reconstructed_legit, _ = model(x_legit_tensor)
        train_legit_mse = torch.mean((reconstructed_legit - x_legit_tensor) ** 2, dim=1).cpu().numpy()

    # Determine Anomaly Threshold (98th percentile of legitimate errors)
    anomaly_threshold = float(np.percentile(train_legit_mse, 98.0))
    print(f"[AUTOENCODER] Established Deep Anomaly Threshold (98th %ile): {anomaly_threshold:.6f}")

    # 4. Evaluate Anomaly Detection Capability on Test Set
    with torch.no_grad():
        x_test_tensor = torch.from_numpy(X_test_all).to(device)
        reconstructed_test, _ = model(x_test_tensor)
        test_mse = torch.mean((reconstructed_test - x_test_tensor) ** 2, dim=1).cpu().numpy()

    # Min-Max normalize reconstruction error into 0.0 to 1.0 deep anomaly score
    norm_deep_score = np.clip((test_mse - train_legit_mse.min()) / (anomaly_threshold * 2 - train_legit_mse.min() + 1e-8), 0.0, 1.0)

    metrics_test = compute_fraud_metrics(y_test, norm_deep_score, threshold=0.5)
    print_metrics_summary("Autoencoder Deep Anomaly Detector (Test Set)", metrics_test)

    # 5. Save Model Checkpoint
    model_path = os.path.join(MODEL_DIR, "autoencoder.pt")
    torch.save({
        "state_dict": model.state_dict(),
        "input_dim": input_dim,
        "anomaly_threshold": anomaly_threshold,
        "training_epochs": epochs
    }, model_path)

    joblib.dump(scaler, os.path.join(MODEL_DIR, "autoencoder_scaler.joblib"))

    print(f"[SAVE] Saved PyTorch Autoencoder checkpoint -> {model_path}")

    return {
        "model": model,
        "anomaly_threshold": anomaly_threshold,
        "test_metrics": metrics_test,
        "test_deep_scores": norm_deep_score
    }

if __name__ == "__main__":
    train_autoencoder()
