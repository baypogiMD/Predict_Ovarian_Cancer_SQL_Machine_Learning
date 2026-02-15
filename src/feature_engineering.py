"""
Clinical feature engineering module
for Ovarian Cancer Risk Prediction.

Adds:
- Log-transformed tumor markers
- Neutrophil-to-Lymphocyte Ratio (NLR)
- Age × Menopause interaction
- Safe numeric transformations
"""

import numpy as np
import pandas as pd
from typing import List

from .config import (
    LOG_TRANSFORM_TUMOR_MARKERS,
    INCLUDE_NLR,
    INCLUDE_INTERACTION_TERM
)


# ======================================================
# Utility: Safe Log Transform
# ======================================================

def safe_log_transform(series: pd.Series) -> pd.Series:
    """
    Apply log1p transformation safely to handle skewed biomarkers.
    """
    return np.log1p(series.clip(lower=0))


# ======================================================
# Tumor Marker Log Transform
# ======================================================

def add_log_transformed_markers(
    df: pd.DataFrame,
    markers: List[str] = ["ca125", "he4", "cea", "afp", "ca199", "ca724"]
) -> pd.DataFrame:
    """
    Add log-transformed tumor marker features.
    """
    df = df.copy()

    for marker in markers:
        if marker in df.columns:
            df[f"log_{marker}"] = safe_log_transform(df[marker])

    return df


# ======================================================
# Neutrophil-Lymphocyte Ratio
# ======================================================

def add_nlr_feature(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Neutrophil-to-Lymphocyte Ratio (NLR).
    """
    df = df.copy()

    if "neu_pct" in df.columns and "lym_pct" in df.columns:
        df["nlr"] = df["neu_pct"] / df["lym_pct"].replace(0, np.nan)

    return df


# ======================================================
# Age × Menopause Interaction
# ======================================================

def add_interaction_terms(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add clinically relevant interaction features.
    """
    df = df.copy()

    if "age" in df.columns and "menopause" in df.columns:
        df["age_menopause_interaction"] = df["age"] * df["menopause"]

    return df


# ======================================================
# Feature Engineering Pipeline
# ======================================================

def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering function.
    Applies all enabled clinical transformations.
    """

    df = df.copy()

    if LOG_TRANSFORM_TUMOR_MARKERS:
        df = add_log_transformed_markers(df)

    if INCLUDE_NLR:
        df = add_nlr_feature(df)

    if INCLUDE_INTERACTION_TERM:
        df = add_interaction_terms(df)

    return df
