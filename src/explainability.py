"""
Explainability module for
Ovarian Cancer Risk Prediction.

Provides:
- SHAP value computation
- Global feature importance
- Summary plots
- Individual prediction explanations
"""

import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any


# ======================================================
# Compute SHAP Values
# ======================================================

def compute_shap_values(
    model: Any,
    X: pd.DataFrame
):
    """
    Compute SHAP values for tree-based models.
    Compatible with CatBoost and Random Forest.
    """

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    return explainer, shap_values


# ======================================================
# Global SHAP Summary Plot
# ======================================================

def plot_shap_summary(
    shap_values,
    X: pd.DataFrame
) -> None:
    """
    Generate SHAP summary (beeswarm) plot.
    """
    shap.summary_plot(shap_values, X)


# ======================================================
# Global SHAP Bar Plot (Clinical Importance)
# ======================================================

def plot_shap_bar(
    shap_values,
    X: pd.DataFrame
) -> None:
    """
    Generate SHAP feature importance bar plot.
    """
    shap.summary_plot(shap_values, X, plot_type="bar")


# ======================================================
# Individual Patient Explanation
# ======================================================

def plot_individual_explanation(
    explainer,
    shap_values,
    X: pd.DataFrame,
    patient_index: int = 0
) -> None:
    """
    Generate SHAP force plot for a specific patient.
    """

    shap.force_plot(
        explainer.expected_value,
        shap_values[patient_index],
        X.iloc[patient_index],
        matplotlib=True
    )


# ======================================================
# Ranked Feature Importance DataFrame
# ======================================================

def get_shap_importance_dataframe(
    shap_values,
    X: pd.DataFrame
) -> pd.DataFrame:
    """
    Return DataFrame of mean absolute SHAP values.
    Useful for reporting tables.
    """

    importance = np.abs(shap_values).mean(axis=0)

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": importance
    }).sort_values("mean_abs_shap", ascending=False)

    return importance_df


# ======================================================
# Save SHAP Plot to File
# ======================================================

def save_shap_summary_plot(
    shap_values,
    X: pd.DataFrame,
    filepath: str
) -> None:
    """
    Save SHAP summary plot to disk.
    """

    shap.summary_plot(shap_values, X, show=False)
    plt.tight_layout()
    plt.savefig(filepath)
    plt.close()
