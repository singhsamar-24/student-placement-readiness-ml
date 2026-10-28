from __future__ import annotations

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline

from src.data import feature_columns
from src.features import add_engineered_features, engineered_numeric_features
from src.train import build_preprocessor


def run_clustering(df: pd.DataFrame, k: int = 4) -> dict[str, object]:
    work = add_engineered_features(df.copy())
    base_numeric, categorical = feature_columns(work)
    numeric = base_numeric + engineered_numeric_features(list(work.columns))
    features = numeric + categorical
    preprocessor = build_preprocessor(numeric, categorical)
    matrix = preprocessor.fit_transform(work[features])
    scores = []
    inertias = []
    for candidate in range(2, 8):
        km = KMeans(n_clusters=candidate, n_init=10, random_state=42)
        labels = km.fit_predict(matrix)
        scores.append({"k": candidate, "silhouette": round(float(silhouette_score(matrix, labels)), 4)})
        inertias.append({"k": candidate, "inertia": round(float(km.inertia_), 2)})
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    labels = model.fit_predict(matrix)
    work["cluster"] = labels
    profiles = (
        work.groupby("cluster")[numeric]
        .mean()
        .round(2)
        .assign(student_count=work.groupby("cluster").size())
        .reset_index()
    )
    descriptions = []
    for _, row in profiles.iterrows():
        strengths = row.drop(labels=["cluster", "student_count"]).sort_values(ascending=False).head(3).index.tolist()
        descriptions.append(
            {
                "cluster": int(row["cluster"]),
                "student_count": int(row["student_count"]),
                "description": "Relatively stronger in " + ", ".join(strengths).replace("_", " "),
            }
        )
    return {
        "data": work,
        "profiles": profiles,
        "silhouette": scores,
        "elbow": inertias,
        "descriptions": descriptions,
    }

