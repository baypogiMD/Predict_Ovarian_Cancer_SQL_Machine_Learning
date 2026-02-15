"""
Preprocessing utilities for Ovarian Cancer Risk Prediction project.

Provides:
- Train/test splitting
- Feature scaling
- Missing value imputation
- Target separation
"""

from typing import Tuple
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from .config import (
    RANDOM_STATE,
    TEST_SIZE
)


# ======================================================
# Feature / Target Split
# ======================================================

def split_features_target(
    df: pd.DataFrame,
    target_column: str = "type"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and target variable.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str

    Returns
    -------
    X : pd.DataFrame
    y : pd.Series
    """
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame.")

    X = df.drop(columns=[target_column])
    y = df[target_column]

    return X, y


# ======================================================
# Train / Test Split
# ======================================================

def train_test_split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    stratify: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Perform stratified train/test split.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y if stratify else None
    )


# ======================================================
# Missing Value Imputation
# ======================================================

def impute_missing_values(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    strategy: str = "median"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Impute missing values using training statistics.

    Parameters
    ----------
    strategy : str
        'mean', 'median', or 'most_frequent'

    Returns
    -------
    X_train_imputed, X_test_imputed
    """
    imputer = SimpleImputer(strategy=strategy)

    X_train_imputed = pd.DataFrame(
        imputer.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )

    X_test_imputed = pd.DataFrame(
        imputer.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return X_train_imputed, X_test_imputed


# ======================================================
# Feature Scaling
# ======================================================

def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standard scale features (for linear models).

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, scaler


# ======================================================
# Full Preprocessing Pipeline
# ======================================================

def full_preprocessing_pipeline(
    df: pd.DataFrame,
    target_column: str = "type",
    scale: bool = False
):
    """
    End-to-end preprocessing pipeline.

    Steps:
    1. Split features and target
    2. Train/test split
    3. Impute missing values
    4. Optional scaling

    Returns
    -------
    X_train, X_test, y_train, y_test
    """

    # Split features and target
    X, y = split_features_target(df, target_column)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)

    # Impute missing values
    X_train, X_test = impute_missing_values(X_train, X_test)

    # Optional scaling
    if scale:
        X_train, X_test, _ = scale_features(X_train, X_test)

    return X_train, X_test, y_train, y_test
