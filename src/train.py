from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from src.config import CONFIG, MODEL_DIR
from src.data import feature_columns, load_dataset, validate_dataset
from src.features import add_engineered_features, engineered_numeric_features

LOGGER = logging.getLogger(__name__)


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [("num", numeric_pipeline, numeric), ("cat", categorical_pipeline, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def candidate_models() -> dict[str, tuple[object, dict[str, list[object]]]]:
    return {
        "Logistic Regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=CONFIG.random_state),
            {"model__C": [0.5, 1.0, 2.0]},
        ),
        "Decision Tree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=CONFIG.random_state),
            {"model__max_depth": [3, 5, 7], "model__min_samples_leaf": [5, 12]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=CONFIG.random_state, n_jobs=-1),
            {"model__n_estimators": [120], "model__max_depth": [5, 8, None], "model__min_samples_leaf": [3, 8]},
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=CONFIG.random_state),
            {"model__n_estimators": [80, 120], "model__learning_rate": [0.04, 0.08], "model__max_depth": [2, 3]},
        ),
        "K-Nearest Neighbors": (
            KNeighborsClassifier(),
            {"model__n_neighbors": [7, 11, 15], "model__weights": ["uniform", "distance"]},
        ),
        "Support Vector Machine": (
            CalibratedClassifierCV(
                estimator=SVC(class_weight="balanced", random_state=CONFIG.random_state),
                method="sigmoid",
                cv=3,
            ),
            {"model__estimator__C": [0.5, 1.0], "model__estimator__kernel": ["rbf", "linear"]},
        ),
    }


def _classification_metrics(model: Pipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
    preds = model.predict(x_test)
    probs = model.predict_proba(x_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    return {
        "accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
    }


def train_classification_models(df: pd.DataFrame) -> dict[str, object]:
    df = add_engineered_features(df)
    base_numeric, categorical = feature_columns(df)
    numeric = base_numeric + engineered_numeric_features(list(df.columns))
    features = numeric + categorical
    x = df[features]
    y = df[CONFIG.target].astype(int)
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=CONFIG.test_size, stratify=y, random_state=CONFIG.random_state
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=CONFIG.random_state)
    results: dict[str, object] = {}
    trained: dict[str, Pipeline] = {}
    for name, (estimator, grid) in candidate_models().items():
        pipeline = Pipeline([("preprocess", build_preprocessor(numeric, categorical)), ("model", estimator)])
        search = GridSearchCV(pipeline, grid, cv=cv, scoring=CONFIG.model_metric, n_jobs=-1)
        search.fit(x_train, y_train)
        best = search.best_estimator_
        metrics = _classification_metrics(best, x_test, y_test)
        metrics["cv_f1"] = round(float(search.best_score_), 4)
        metrics["best_params"] = search.best_params_
        results[name] = metrics
        trained[name] = best

    voting = VotingClassifier(
        estimators=[
            ("lr", trained["Logistic Regression"]),
            ("rf", trained["Random Forest"]),
            ("gb", trained["Gradient Boosting"]),
        ],
        voting="soft",
    )
    voting.fit(x_train, y_train)
    results["Voting Ensemble"] = _classification_metrics(voting, x_test, y_test)
    results["Voting Ensemble"]["cv_f1"] = None
    trained["Voting Ensemble"] = voting

    best_name = max(results, key=lambda key: (results[key]["f1"], results[key]["roc_auc"]))
    return {
        "best_model_name": best_name,
        "best_model": trained[best_name],
        "all_models": trained,
        "metrics": results,
        "features": {"numeric": numeric, "categorical": categorical, "all": features},
        "class_distribution": y.value_counts(normalize=False).sort_index().to_dict(),
        "train_shape": x_train.shape,
        "test_shape": x_test.shape,
        "x_test": x_test,
        "y_test": y_test,
    }


def train_regression_if_supported(df: pd.DataFrame, features: list[str], numeric: list[str], categorical: list[str]) -> dict[str, object] | None:
    if CONFIG.regression_target not in df.columns or df[CONFIG.regression_target].max() <= 0:
        return None
    reg_df = add_engineered_features(df[df[CONFIG.regression_target] > 0].copy())
    if len(reg_df) < 50:
        return None
    x = reg_df[features]
    y = reg_df[CONFIG.regression_target].astype(float)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=CONFIG.random_state)
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0, random_state=CONFIG.random_state),
    }
    metrics = {}
    trained = {}
    for name, estimator in models.items():
        pipeline = Pipeline([("preprocess", build_preprocessor(numeric, categorical)), ("model", estimator)])
        pipeline.fit(x_train, y_train)
        preds = pipeline.predict(x_test)
        metrics[name] = {
            "mae": round(float(mean_absolute_error(y_test, preds)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, preds))), 4),
            "r2": round(float(r2_score(y_test, preds)), 4),
        }
        trained[name] = pipeline
    best_name = max(metrics, key=lambda key: metrics[key]["r2"])
    return {"best_regression_model_name": best_name, "metrics": metrics, "model": trained[best_name]}


def train_and_persist() -> dict[str, object]:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    df = load_dataset()
    validation = validate_dataset(df)
    if not validation["is_valid"]:
        raise ValueError(f"Dataset validation failed: {validation}")
    classification = train_classification_models(df)
    regression = train_regression_if_supported(
        df,
        classification["features"]["all"],
        classification["features"]["numeric"],
        classification["features"]["categorical"],
    )
    model_bundle = {
        "model": classification["best_model"],
        "model_name": classification["best_model_name"],
        "features": classification["features"],
        "target": CONFIG.target,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(CONFIG.dataset_path),
    }
    joblib.dump(model_bundle, MODEL_DIR / "placement_model.joblib")
    if regression:
        joblib.dump(regression, MODEL_DIR / "salary_regression.joblib")

    metadata = {
        "model_name": classification["best_model_name"],
        "created_at": model_bundle["created_at"],
        "metrics": classification["metrics"],
        "features": classification["features"],
        "class_distribution": classification["class_distribution"],
        "train_shape": classification["train_shape"],
        "test_shape": classification["test_shape"],
        "selection_metric": CONFIG.model_metric,
        "dataset": "Demonstration/Synthetic Dataset",
        "validation": validation,
        "regression": regression and {"best_model": regression["best_regression_model_name"], "metrics": regression["metrics"]},
    }
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def load_model_bundle() -> dict[str, object]:
    path = MODEL_DIR / "placement_model.joblib"
    if not path.exists():
        train_and_persist()
    return joblib.load(path)

