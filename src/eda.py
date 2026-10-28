from __future__ import annotations

import pandas as pd


def dataset_summary(df: pd.DataFrame) -> dict[str, object]:
    numeric = df.select_dtypes(include="number")
    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    outliers = ((numeric < (q1 - 1.5 * iqr)) | (numeric > (q3 + 1.5 * iqr))).sum().sort_values(ascending=False)
    return {
        "shape": df.shape,
        "feature_types": df.dtypes.astype(str).to_dict(),
        "missing_values": df.isna().sum().sort_values(ascending=False).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "outlier_counts": outliers.to_dict(),
        "statistics": df.describe(include="all").fillna("").astype(str),
    }

