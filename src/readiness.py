from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import CONFIG


def _score_feature(name: str, value: float) -> float:
    if pd.isna(value):
        return 0.0
    if name == "cgpa":
        return np.clip((value - 4.0) / 6.0 * 100, 0, 100)
    if name == "backlogs":
        return np.clip(100 - value * 22, 0, 100)
    if name in {"projects", "certifications"}:
        return np.clip(value / 4 * 100, 0, 100)
    if name in {"internships", "hackathons"}:
        return np.clip(value / 3 * 100, 0, 100)
    return np.clip(value, 0, 100)


def calculate_readiness(row: pd.Series | dict) -> dict[str, object]:
    series = pd.Series(row)
    breakdown: dict[str, float] = {}
    for group, features in CONFIG.readiness_groups.items():
        scores = [_score_feature(feature, float(series.get(feature, np.nan))) for feature in features]
        breakdown[group] = round(float(np.mean(scores)), 1) if scores else 0.0
    overall = round(float(np.mean(list(breakdown.values()))), 1)
    if overall >= 75:
        category = "High Readiness"
    elif overall >= 55:
        category = "Medium Readiness"
    else:
        category = "Needs Improvement"
    return {"score": overall, "category": category, "breakdown": breakdown}

