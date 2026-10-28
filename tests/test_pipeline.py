from __future__ import annotations

import pandas as pd

from src.data import generate_synthetic_dataset, validate_dataset
from src.predict import predict_student
from src.readiness import calculate_readiness
from src.recommendations import generate_recommendations, weak_features
from src.train import train_classification_models


def test_dataset_generation_and_validation(tmp_path):
    df = generate_synthetic_dataset(tmp_path / "demo.csv", rows=120)
    validation = validate_dataset(df)
    assert validation["is_valid"]
    assert df["placed"].isin([0, 1]).all()
    assert len(df) == 120


def test_training_pipeline_returns_metrics(tmp_path):
    df = generate_synthetic_dataset(tmp_path / "demo.csv", rows=180)
    result = train_classification_models(df)
    assert result["best_model_name"] in result["metrics"]
    best_metrics = result["metrics"][result["best_model_name"]]
    assert 0 <= best_metrics["f1"] <= 1
    assert 0 <= best_metrics["roc_auc"] <= 1


def test_prediction_readiness_and_recommendations():
    row = {
        "branch": "Computer",
        "specialization": "Software",
        "cgpa": 6.2,
        "attendance_pct": 68.0,
        "backlogs": 2,
        "coding_score": 48.0,
        "dsa_score": 42.0,
        "aptitude_score": 58.0,
        "communication_score": 55.0,
        "interview_score": 52.0,
        "projects": 1,
        "certifications": 0,
        "internships": 0,
        "hackathons": 0,
    }
    readiness = calculate_readiness(pd.Series(row))
    recommendations = generate_recommendations(row)
    weak = weak_features(row)
    assert 0 <= readiness["score"] <= 100
    assert recommendations
    assert weak
    prediction = predict_student(row)
    assert 0 <= prediction["placement_probability"] <= 100
    assert prediction["recommendations"]

