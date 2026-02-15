"""
Model evaluation module
for Ovarian Cancer Risk Prediction.

Provides:
- ROC-AUC
- Confusion matrix
- Classification metrics
- Cross-validation
- Calibration curves
- Threshold optimization
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any

from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    brier_score_loss,
    precision_recall_curve
)

from sklearn.model_selection import cross_val_score, StratifiedKFold

from .config import (
    PRIMARY_METRIC,
    N_SPLITS_CV,
    RANDOM_STATE
)


# ======================================================
# Basic Evaluation
# ======================================================

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Evaluate model performance.
    """

    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)

    results = {
        "roc_auc": roc_auc_score(y_test, probs),
        "brier_score": brier_score_loss(y_test, probs),
        "confusion_matrix": confusion_matrix(y_test, preds),
        "classification_report": classification_report(
            y_test, preds, output_dict=True
        )
    }

    return results


# ======================================================
# ROC Curve Plot
# ======================================================

def plot_roc_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Plot ROC curve.
    """

    probs = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)

    auc = roc_auc_score(y_test, probs)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve (AUC = {auc:.3f})")
    plt.show()


# ======================================================
# Cross-Validation
# ======================================================

def cross_validate_model(
    model: Any,
    X: pd.DataFrame,
    y: pd.Series
) -> float:
    """
    Perform stratified cross-validation.
    """

    cv = StratifiedKFold(
        n_splits=N_SPLITS_CV,
        shuffle=True,
        random_state=RANDOM_STATE
    )

    scores = cross_val_score(
        model,
        X,
        y,
        cv=cv,
        scoring=PRIMARY_METRIC
    )

    return scores.mean()


# ======================================================
# Calibration Curve
# ======================================================

def plot_calibration_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_bins: int = 10
) -> None:
    """
    Plot calibration curve.
    """

    probs = model.predict_proba(X_test)[:, 1]

    bins = np.linspace(0, 1, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1

    observed = []
    predicted = []

    for i in range(n_bins):
        mask = bin_ids == i
        if np.sum(mask) > 0:
            observed.append(y_test[mask].mean())
            predicted.append(probs[mask].mean())

    plt.figure()
    plt.plot(predicted, observed)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Frequency")
    plt.title("Calibration Curve")
    plt.show()


# ======================================================
# Sensitivity-Optimized Threshold
# ======================================================

def optimize_threshold_for_sensitivity(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    min_sensitivity: float = 0.95
) -> float:
    """
    Find lowest threshold achieving desired sensitivity.
    """

    probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0, 1, 200)

    for t in thresholds:
        preds = (probs >= t).astype(int)
        cm = confusion_matrix(y_test, preds)

        # Sensitivity = TP / (TP + FN)
        tp = cm[1, 1]
        fn = cm[1, 0]

        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0

        if sensitivity >= min_sensitivity:
            return t

    return 0.5


# ======================================================
# Precision-Recall Curve
# ======================================================

def plot_precision_recall_curve(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """
    Plot precision-recall curve.
    """

    probs = model.predict_proba(X_test)[:, 1]
    precision, recall, _ = precision_recall_curve(y_test, probs)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.show()
