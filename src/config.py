"""
Configuration file for Ovarian Cancer Risk Prediction Project.

Centralizes:
- File paths
- Random state control
- Model hyperparameters
- Cross-validation settings
- Clinical risk thresholds
"""

from pathlib import Path

# ======================================================
# Project Paths
# ======================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATABASE_DIR = DATA_DIR / "database"

DATABASE_PATH = DATABASE_DIR / "ovarian_cancer.db"

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "models"

# Create directories if they don't exist
for directory in [
    DATA_DIR,
    RAW_DATA_DIR,
    PROCESSED_DATA_DIR,
    DATABASE_DIR,
    OUTPUT_DIR,
    FIGURE_DIR,
    MODEL_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)


# ======================================================
# Reproducibility
# ======================================================

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_SPLITS_CV = 5


# ======================================================
# Logistic Regression Settings
# ======================================================

LOGISTIC_PARAMS = {
    "max_iter": 2000,
    "solver": "lbfgs",
    "random_state": RANDOM_STATE
}


# ======================================================
# Random Forest Settings
# ======================================================

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": None,
    "random_state": RANDOM_STATE
}


# ======================================================
# CatBoost Settings (Primary Model)
# ======================================================

CATBOOST_PARAMS = {
    "iterations": 800,
    "depth": 6,
    "learning_rate": 0.05,
    "loss_function": "Logloss",
    "eval_metric": "AUC",
    "random_seed": RANDOM_STATE,
    "verbose": False
}


# ======================================================
# Clinical Risk Thresholds
# ======================================================

LOW_RISK_THRESHOLD = 0.25
INTERMEDIATE_RISK_THRESHOLD = 0.60


# ======================================================
# Feature Engineering Options
# ======================================================

LOG_TRANSFORM_TUMOR_MARKERS = True
INCLUDE_NLR = True
INCLUDE_INTERACTION_TERM = True


# ======================================================
# Evaluation Metrics
# ======================================================

PRIMARY_METRIC = "roc_auc"
SECONDARY_METRICS = ["precision", "recall", "f1"]


# ======================================================
# Version
# ======================================================

PROJECT_VERSION = "1.0.0"
