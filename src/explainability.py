from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import CONFIG
from src.features import add_engineered_features
from src.train import load_model_bundle


def global_feature_importance(df: pd.DataFrame, limit: int = 15) -> pd.DataFrame:
    bundle = load_model_bundle()
    model = bundle["model"]
    features = bundle["features"]["all"]
    work = add_engineered_features(df.copy())
    x = work[features]
    y = work[CONFIG.target].astype(int)
    result = permutation_importance(model, x, y, n_repeats=5, random_state=CONFIG.random_state, scoring="f1")
    importance = pd.DataFrame({"feature": features, "importance": result.importances_mean})
    return importance.sort_values("importance", ascending=False).head(limit)


def local_contributors(input_row: dict, limit: int = 5) -> dict[str, list[tuple[str, float]]]:
    bundle = load_model_bundle()
    model = bundle["model"]
    features = [f for f in bundle["features"]["all"] if f in input_row]
    base = pd.Series(input_row)
    rows = []
    for feature in features:
        variant = base.copy()
        value = variant[feature]
        if pd.api.types.is_number(value):
            variant[feature] = max(0, float(value) * 0.75)
        else:
            continue
        original_prob = model.predict_proba(add_engineered_features(pd.DataFrame([base]))[bundle["features"]["all"]])[:, 1][0]
        variant_prob = model.predict_proba(add_engineered_features(pd.DataFrame([variant]))[bundle["features"]["all"]])[:, 1][0]
        rows.append((feature, float(original_prob - variant_prob)))
    ordered = sorted(rows, key=lambda item: item[1], reverse=True)
    positive = [(name, round(score * 100, 2)) for name, score in ordered if score > 0][:limit]
    improvement = [(name, round(abs(score) * 100, 2)) for name, score in ordered[::-1] if score < 0][:limit]
    return {"positive": positive, "improvement": improvement}

