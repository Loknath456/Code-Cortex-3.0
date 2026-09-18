"""
FIN-XR Metric Calculations for Imbalanced Fraud Detection
Focuses on PR-AUC, Precision, Recall, F1, ROC-AUC, and Confusion Matrix.
"""

import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    precision_recall_curve, auc, roc_auc_score, confusion_matrix
)

def compute_fraud_metrics(y_true, y_pred_prob, threshold=0.5):
    """
    Computes comprehensive classification metrics for imbalanced datasets.
    """
    y_pred = (y_pred_prob >= threshold).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    # Precision-Recall Curve & PR-AUC
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_pred_prob)
    pr_auc = auc(rec_curve, prec_curve)

    # ROC-AUC
    try:
        roc_auc = roc_auc_score(y_true, y_pred_prob)
    except ValueError:
        roc_auc = 0.5

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "pr_auc": float(pr_auc),
        "roc_auc": float(roc_auc),
        "confusion_matrix": {
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp)
        },
        "threshold_used": float(threshold)
    }

def print_metrics_summary(model_name, metrics):
    """
    Prints clean tabular evaluation summary.
    """
    print(f"\n==========================================")
    print(f" EVALUATION METRICS: {model_name}")
    print(f"==========================================")
    print(f" PR-AUC    : {metrics['pr_auc']:.4f}  <-- Primary Metric for Imbalance")
    print(f" ROC-AUC   : {metrics['roc_auc']:.4f}")
    print(f" Precision : {metrics['precision']:.4f}")
    print(f" Recall    : {metrics['recall']:.4f}")
    print(f" F1 Score  : {metrics['f1_score']:.4f}")
    cm = metrics['confusion_matrix']
    print(f" Confusion Matrix:")
    print(f"   TN: {cm['true_negatives']} | FP: {cm['false_positives']}")
    print(f"   FN: {cm['false_negatives']} | TP: {cm['true_positives']}")
    print(f"==========================================\n")
