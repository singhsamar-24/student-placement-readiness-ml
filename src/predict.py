from __future__ import annotations

import pandas as pd

from src.features import add_engineered_features
from src.readiness import calculate_readiness
from src.recommendations import generate_recommendations
from src.train import load_model_bundle


def predict_student(input_row: dict) -> dict[str, object]:
    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]["all"]
    df = add_engineered_features(pd.DataFrame([input_row]))
    probability = float(model.predict_proba(df[features])[:, 1][0])
    prediction = int(probability >= 0.5)
    confidence = "High" if probability >= 0.75 or probability <= 0.25 else "Medium" if abs(probability - 0.5) >= 0.12 else "Low"
    return {
        "prediction": prediction,
        "label": "Likely Placed" if prediction else "Needs Intervention",
        "placement_probability": round(probability * 100, 2),
        "confidence": confidence,
        "readiness": calculate_readiness(input_row),
        "recommendations": generate_recommendations(input_row),
    }

