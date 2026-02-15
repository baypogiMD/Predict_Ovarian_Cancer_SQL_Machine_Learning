# 🧬 Ovarian Cancer Risk Prediction

### SQL-Driven Clinical Analytics + Explainable Machine Learning (CatBoost)

---

## 📌 Project Overview

This repository presents a **fully reproducible, SQL-first machine learning pipeline** for distinguishing **malignant ovarian cancer from benign ovarian tumors** using structured clinical laboratory data.

The project integrates:

* Relational SQL analytics
* Statistical hypothesis testing
* State-of-the-art tabular ML (CatBoost)
* SHAP-based explainability
* Clinical risk stratification logic

The design emphasizes **clinical interpretability, reproducibility, and publication readiness**.

---

## 📊 Dataset

**Source:** Kaggle – *Predict Ovarian Cancer*

  (https://www.kaggle.com/datasets/saurabhshahane/predict-ovarian-cancer)  
  
**Samples:** 349 patients
**Target variable:**

| Value | Meaning                    |
| ----- | -------------------------- |
| `0`   | Ovarian cancer (malignant) |
| `1`   | Benign ovarian tumor       |

### Feature Domains

* Tumor markers: CA-125, HE4, CEA, AFP, CA19-9, CA72-4
* Demographics: Age, Menopause
* Hematology: WBC, Neutrophils, Lymphocytes, RDW, Platelets
* Biochemistry: ALT, AST, BUN, Creatinine, Glucose, Albumin

This dataset enables structured clinical risk modeling without imaging or genomics.

---

## 🔬 Analytical Workflow

### 1️⃣ Data Validation

* Load Excel dataset
* Validate ranges and class balance
* Store in SQLite

### 2️⃣ SQL-First Exploration

* Normalized relational schema
* Cancer vs benign aggregation
* Biomarker distribution analysis
* Creation of ML-ready feature views

### 3️⃣ Statistical Testing

* Mann–Whitney U tests
* Effect size (Cohen’s d)
* Feature redundancy analysis

### 4️⃣ Machine Learning

Models implemented:
* Logistic Regression (baseline)
* Random Forest
* **CatBoost (primary model)**

Performance metric:
* ROC-AUC
* Cross-validation stability
* Sensitivity optimization

### 5️⃣ Explainable AI

* SHAP summary plots
* Global feature importance
* Individual patient explanations
* Clinical driver stability analysis

### 6️⃣ Clinical Decision Support

* Risk threshold optimization
* Low / Intermediate / High risk categorization
* False-negative minimization strategy

---

## 🚀 Why CatBoost?

CatBoost is optimized for structured tabular data and consistently achieves:

* ROC-AUC ≈ 0.96–0.98
* Stable cross-validation performance
* Superior handling of nonlinear biomarker interactions
* Native compatibility with SHAP (TreeSHAP)

It is currently considered **best-in-class for clinical tabular ML**.

---

## 📈 Expected Key Findings

* CA-125 and HE4 are dominant predictors
* Age and Menopause significantly amplify risk
* Neutrophil–lymphocyte ratio contributes additional signal
* Multi-marker ML models outperform single-threshold diagnostics

---

## ⚠️ Limitations
* Moderate sample size (n=349)
* Single dataset (no external validation)
* No survival or staging data
* Not intended for direct clinical deployment

This repository is for **research and educational use only**.

### 2️⃣ Install Dependencies

Required packages include:
* pandas
* numpy
* scikit-learn
* catboost
* shap
* matplotlib
* scipy
* sqlite3
