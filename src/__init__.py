"""
Ovarian Cancer Risk Prediction
SQL-Driven Clinical Machine Learning Package

This package provides utilities for:
- Database connectivity
- Data preprocessing
- Clinical feature engineering
- Machine learning modeling
- Model evaluation
- Explainable AI (SHAP)

Author: Your Name
Project: ovarian-cancer-risk-prediction-sql-ml
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# ===============================
# Core Module Imports
# ===============================

from .config import (
    DATABASE_PATH,
    RANDOM_STATE
)

from .db_utils import (
    get_connection,
    load_model_dataset
)

from .preprocessing import (
    train_test_split_data,
    scale_features
)

from .feature_engineering import (
    add_clinical_features
)

from .modeling import (
    train_logistic_regression,
    train_random_forest,
    train_catboost
)

from .evaluation import (
    evaluate_model,
    cross_validate_model
)

from .explainability import (
    compute_shap_values,
    plot_shap_summary
)

# ===============================
# Public API
# ===============================

__all__ = [
    "DATABASE_PATH",
    "RANDOM_STATE",
    "get_connection",
    "load_model_dataset",
    "train_test_split_data",
    "scale_features",
    "add_clinical_features",
    "train_logistic_regression",
    "train_random_forest",
    "train_catboost",
    "evaluate_model",
    "cross_validate_model",
    "compute_shap_values",
    "plot_shap_summary",
]
