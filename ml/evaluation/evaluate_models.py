"""
FIN-XR Comprehensive Model Evaluation & Visualization Suite
Generates all 12 required plots and exports model_metadata.json.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix

PLOTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "plots"))
METRICS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "outputs", "metrics"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)

# Styling Config
plt.style.use("dark_background")
COLORS = ["#20E6C2", "#3B82F6", "#F97316", "#EF4444", "#10B981"]

def generate_all_evaluation_plots(y_test, model_predictions_dict, bank_scored_df=None):
    """
    Generates all 12 required visualizations.
    """
    print("\n==================================================")
    print(" GENERATING VISUALIZATIONS & METADATA REPORT")
    print("==================================================")

    # 1. Class Distribution Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x=y_test, palette=["#20E6C2", "#EF4444"], ax=ax)
    ax.set_title("Credit Card Target Class Distribution (0=Legit, 1=Fraud)", fontsize=10, color="white")
    ax.set_xticklabels(["Legitimate (0)", "Fraud (1)"])
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "class_distribution.png"), dpi=200)
    plt.close()
    print(" Saved plot: class_distribution.png")

    # 2, 3, 4. Confusion Matrices
    for name, probs in model_predictions_dict.items():
        y_pred = (probs >= 0.5).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(f"Confusion Matrix: {name.upper()}", fontsize=10, color="white")
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, f"confusion_matrix_{name}.png"), dpi=200)
        plt.close()
        print(f" Saved plot: confusion_matrix_{name}.png")

    # 5. Precision-Recall Curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, probs) in enumerate(model_predictions_dict.items()):
        prec, rec, _ = precision_recall_curve(y_test, probs)
        ax.plot(rec, prec, label=name.upper(), color=COLORS[i % len(COLORS)], linewidth=2)

    ax.set_title("Precision-Recall Curves (Imbalanced Fraud Detection)", fontsize=11, color="white")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "precision_recall_curves.png"), dpi=200)
    plt.close()
    print(" Saved plot: precision_recall_curves.png")

    # 6. ROC Curves
    fig, ax = plt.subplots(figsize=(7, 5))
    for i, (name, probs) in enumerate(model_predictions_dict.items()):
        fpr, tpr, _ = roc_curve(y_test, probs)
        ax.plot(fpr, tpr, label=name.upper(), color=COLORS[i % len(COLORS)], linewidth=2)

    ax.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    ax.set_title("ROC Curves", fontsize=11, color="white")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.15)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curves.png"), dpi=200)
    plt.close()
    print(" Saved plot: roc_curves.png")

    # 7. Model Comparison Bar Chart (PR-AUC)
    model_names = list(model_predictions_dict.keys())
    pr_aucs = []
    from evaluation.metrics import compute_fraud_metrics
    for name, probs in model_predictions_dict.items():
        m = compute_fraud_metrics(y_test, probs)
        pr_aucs.append(m['pr_auc'])

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([n.upper() for n in model_names], pr_aucs, color=["#3B82F6", "#F97316", "#20E6C2", "#EF4444"])
    ax.set_title("Model Comparison: PR-AUC Score (Higher is Better)", fontsize=11, color="white")
    ax.set_ylabel("PR-AUC Score")
    ax.set_ylim(0, 1.05)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f"{yval:.4f}", ha='center', va='bottom', color='white', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "model_comparison.png"), dpi=200)
    plt.close()
    print(" Saved plot: model_comparison.png")

    # 8 & 9. Feature Importances for XGBoost & Random Forest
    for model_name in ["xgb", "rf"]:
        imp_path = os.path.join(METRICS_DIR, f"{model_name}_feature_importance.csv")
        if os.path.exists(imp_path):
            s_imp = pd.read_csv(imp_path, index_col=0).iloc[:, 0].head(10)
            fig, ax = plt.subplots(figsize=(7, 4))
            s_imp.plot(kind="barh", color="#20E6C2" if model_name=="xgb" else "#3B82F6", ax=ax)
            ax.invert_yaxis()
            ax.set_title(f"Top 10 Feature Importances ({model_name.upper()})", fontsize=10, color="white")
            plt.tight_layout()
            fig.savefig(os.path.join(PLOTS_DIR, f"{model_name}_feature_importance.png"), dpi=200)
            plt.close()
            print(f" Saved plot: {model_name}_feature_importance.png")

    # 10. Autoencoder Reconstruction Distribution
    if "autoencoder" in model_predictions_dict:
        fig, ax = plt.subplots(figsize=(7, 4))
        probs = model_predictions_dict["autoencoder"]
        sns.kdeplot(probs[y_test == 0], label="Legitimate (Class 0)", color="#20E6C2", fill=True, ax=ax)
        sns.kdeplot(probs[y_test == 1], label="Fraud (Class 1)", color="#EF4444", fill=True, ax=ax)
        ax.set_title("Autoencoder Reconstruction Error Distribution", fontsize=10, color="white")
        ax.set_xlabel("Normalized Deep Anomaly Score")
        ax.legend()
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "autoencoder_reconstruction_distribution.png"), dpi=200)
        plt.close()
        print(" Saved plot: autoencoder_reconstruction_distribution.png")

    # 11 & 12. Bank Transaction Anomaly Distribution
    if bank_scored_df is not None and "pattern_status" in bank_scored_df.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x="pattern_status", data=bank_scored_df, palette=["#20E6C2", "#F59E0B", "#F97316", "#EF4444"], ax=ax)
        ax.set_title("Bank Transaction Behavioral Pattern Distribution", fontsize=10, color="white")
        ax.set_ylabel("Count")
        plt.xticks(rotation=15)
        plt.tight_layout()
        fig.savefig(os.path.join(PLOTS_DIR, "bank_anomaly_distribution.png"), dpi=200)
        plt.close()
        print(" Saved plot: bank_anomaly_distribution.png")

def create_model_metadata_json(all_metrics_dict):
    """
    Creates models/model_metadata.json for production FastAPI backend integration.
    """
    metadata = {
        "project_name": "FIN-XR",
        "system_title": "AI-Powered Transaction Pattern Intelligence & Risk Analytics",
        "architecture": {
            "unsupervised_bank_engine": "Isolation Forest",
            "supervised_fraud_baseline": "Logistic Regression",
            "supervised_tree_classifier": "Random Forest",
            "supervised_primary_engine": "XGBoost",
            "deep_learning_engine": "PyTorch Autoencoder (Legitimate Reconstruction)"
        },
        "model_artifacts": {
            "isolation_forest": "models/isolation_forest.joblib",
            "logistic_regression": "models/logistic_regression.joblib",
            "random_forest": "models/random_forest.joblib",
            "xgboost": "models/xgboost.joblib",
            "autoencoder": "models/autoencoder.pt",
            "bank_scaler": "models/bank_scaler.joblib",
            "creditcard_scaler": "models/creditcard_scaler.joblib"
        },
        "pattern_status_thresholds": {
            "normal": "0.00 - 0.39",
            "minor_deviation": "0.40 - 0.59",
            "behavior_shift": "0.60 - 0.79",
            "high_deviation": "0.80 - 1.00"
        },
        "metrics": all_metrics_dict
    }

    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"[METADATA] Created model metadata contract -> {meta_path}")
