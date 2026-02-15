"""
Model training module
for Ovarian Cancer Risk Prediction.

Provides:
- Logistic Regression
- Random Forest
- CatBoost
- Model saving utilities
"""

import joblib
import pandas as pd
from typing import Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from catboost import CatBoostClassifier

from .config import (
    LOGISTIC_PARAMS,
    RANDOM_FOREST_PARAMS,
    CATBOOST_PARAMS,
    MODEL_DIR
)


# ======================================================
# Logistic Regression
# ======================================================

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> LogisticRegression:
    """
    Train Logistic Regression model.
    """
    model = LogisticRegression(**LOGISTIC_PARAMS)
    model.fit(X_train, y_train)
    return model


# ======================================================
# Random Forest
# ======================================================

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> RandomForestClassifier:
    """
    Train Random Forest model.
    """
    model = RandomForestClassifier(**RANDOM_FOREST_PARAMS)
    model.fit(X_train, y_train)
    return model


# ======================================================
# CatBoost (Primary Model)
# ======================================================

def train_catboost(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> CatBoostClassifier:
    """
    Train CatBoost classifier.
    """
    model = CatBoostClassifier(**CATBOOST_PARAMS)
    model.fit(X_train, y_train)
    return model


# ======================================================
# Generic Model Trainer
# ======================================================

def train_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Any:
    """
    Unified model training interface.

    model_type options:
    - "logistic"
    - "random_forest"
    - "catboost"
    """

    if model_type == "logistic":
        return train_logistic_regression(X_train, y_train)

    elif model_type == "random_forest":
        return train_random_forest(X_train, y_train)

    elif model_type == "catboost":
        return train_catboost(X_train, y_train)

    else:
        raise ValueError(
            "Invalid model_type. Choose from: "
            "'logistic', 'random_forest', 'catboost'."
        )


# ======================================================
# Model Saving
# ======================================================

def save_model(
    model: Any,
    model_name: str
) -> None:
    """
    Save trained model to outputs/models directory.
    """

    model_path = MODEL_DIR / f"{model_name}.pkl"
    joblib.dump(model, model_path)


# ======================================================
# Model Loading
# ======================================================

def load_model(
    model_name: str
) -> Any:
    """
    Load trained model from outputs/models directory.
    """

    model_path = MODEL_DIR / f"{model_name}.pkl"

    if not model_path.exists():
        raise FileNotFoundError(f"Model '{model_name}' not found.")

    return joblib.load(model_path)
