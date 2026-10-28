# 🎯 Student Placement Prediction & Career Readiness Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![UI: Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![Framework: Scikit--Learn](https://img.shields.io/badge/ML-Scikit--Learn-F7931E.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Academic Project](https://img.shields.io/badge/NIET-ML%20PBL%20Capstone-blueviolet.svg)](docs/)
[![SDGs](https://img.shields.io/badge/SDG-4%20|%208%20|%209-orange.svg)](https://sdgs.un.org/goals)

An explainable Machine Learning decision-support and analytics platform engineered for university training and placement cells. Evaluates student placement likelihood, computes transparent career readiness scores, generates personalized curriculum interventions, and performs student cohort segmentation using unsupervised clustering.

Developed for the **Machine Learning Project-Based Learning (PBL)** curriculum at **Noida Institute of Engineering & Technology (NIET)**.

---

## 🏛️ System Architecture & ML Pipeline

```
[Raw Student Profile & Academic Records]
                    │
                    ▼
[Data Ingestion & Integrity Validation]
    ├── Missing Value & Duplicate Checks
    └── Outlier Detection & Range Bounds
                    │
                    ▼
[Leakage-Safe Preprocessing Pipeline]
    ├── Numerical Features: RobustScaler / StandardScaler
    └── Categorical Features: OneHotEncoder (Branch, Specialization)
                    │
                    ▼
[Feature Engineering & Selection]
    ├── Technical Competency Index (DSA, Coding, Projects)
    ├── Professional Quotient (Aptitude, Communication, Interviews)
    └── Experience Factor (Internships, Hackathons, Certifications)
                    │
                    ▼
         ┌──────────┴──────────┐
         ▼                     ▼
[Supervised Models]     [Unsupervised Clustering]
 ├── Random Forest       └── K-Means Segmentation
 ├── Gradient Boosting       (Silhouette & Inertia Analysis)
 ├── Logistic Regression
 ├── Support Vector Machine
 └── Soft Voting Ensemble
         │
         ▼
[Inference & Decision Support Engine]
 ├── Placement Probability (predict_proba)
 ├── Transparent Readiness Score (0-100)
 ├── Feature Importance & Local Explanations
 └── Personalized Actionable Recommendations
```

---

## ⚡ Machine Learning Models & Capabilities

| Task | Algorithms Evaluated | Primary Selection Metric | Deployed Artifact |
|---|---|---|---|
| **Placement Classification** | Random Forest, Gradient Boosting, SVM, Logistic Regression, KNN, Soft Voting Ensemble | F1-Score (Macro) & ROC-AUC | `models/placement_model.joblib` |
| **Package Regression** | Linear Regression, Ridge Regression | RMSE & MAE ($R^2$) | `models/salary_regression.joblib` |
| **Student Segmentation** | K-Means Clustering | Silhouette Score & Elbow Inertia | Dynamic Cluster Analytics |

---

## 📂 Project Directory Structure

```
student-placement-readiness-ml/
├── app/                                     # Streamlit UI pages & components
├── configs/                                 # Feature configuration YAMLs
├── data/                                    # Datasets
│   ├── raw/student_placement_demo.csv       # Demonstration dataset
│   └── processed/                           # Scaled features & splits
├── docs/                                    # Academic reports & presentations
│   ├── Month_2_Report_ML_PBL.docx           # Mid-term progress report
│   ├── Month_3_Final_Report_ML_PBL.docx     # Comprehensive final report
│   ├── NIET_PBL_ProgressReport_ML (1).docx  # Progress report
│   ├── ML-FILE 1-12.docx                    # Complete lab assignments manual
│   └── ml.pptx                              # Capstone defense presentation
├── models/                                  # Persisted model artifacts & metrics
│   ├── placement_model.joblib               # Trained classifier pipeline
│   ├── salary_regression.joblib             # Package prediction regressor
│   └── metrics.json                         # Cross-validation performance logs
├── notebooks/                               # Exploratory data analysis notebooks
├── src/                                     # Core ML modular source code
│   ├── data.py                              # Data loading & validation
│   ├── eda.py                               # Statistical profiling & correlation
│   ├── features.py                          # Feature engineering & transformers
│   ├── train.py                             # Cross-validation & hyperparameter tuning
│   ├── predict.py                           # Model loading & inference
│   ├── readiness.py                         # Rule-based educational readiness metric
│   ├── recommendations.py                   # Automated gap remediation suggestions
│   ├── explainability.py                    # Feature importance ranking & insights
│   ├── clustering.py                        # K-Means cohort profiling
│   └── config.py                            # Constants & pipeline settings
├── tests/
│   └── test_pipeline.py                     # Pytest pipeline verification suite
├── .gitignore                               # Standard Python ML gitignore
├── LICENSE                                  # MIT License
├── README.md                                # Project documentation
├── requirements.txt                         # Python dependencies
├── scripts_train.py                         # End-to-end training execution script
└── app.py                                   # Streamlit interactive application entry
```

---

## 🚀 Quickstart & Setup Guide

### 1. Prerequisites
- Python 3.10 or 3.11
- Virtual environment recommended

### 2. Environment Setup
```bash
# Clone the repository
git clone git@github.com:singhsamar-24/student-placement-readiness-ml.git
cd student-placement-readiness-ml

# Create virtual environment
python -m venv .venv
# Activate on Windows:
.venv\Scripts\activate
# Activate on Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Training & Evaluation
Train the classification, regression, and clustering pipelines:
```bash
python scripts_train.py
```
This evaluates all algorithms with stratified k-fold cross-validation and saves the top-performing models in `models/`.

### 4. Run the Web Application
Launch the interactive Streamlit dashboard:
```bash
streamlit run app.py
```

### 5. Run the Automated Test Suite
```bash
pytest tests/ -v
```

---

## 👨‍💻 Author & Academic Attribution

- **Developer:** Samar Singh
- **Roll Number:** 2501331940042
- **Programme:** B.Tech Computer Science & Engineering (Artificial Intelligence)
- **Institution:** Noida Institute of Engineering & Technology (NIET), Greater Noida
- **Course:** Machine Learning Capstone (PBL)
- **UN SDG Impact:**
  - **SDG 4:** Quality Education
  - **SDG 8:** Decent Work and Economic Growth
  - **SDG 9:** Industry, Innovation and Infrastructure

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
