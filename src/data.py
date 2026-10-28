from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import CONFIG, DATA_RAW_DIR

LOGGER = logging.getLogger(__name__)


BRANCHES = ["Computer", "IT", "ENTC", "Mechanical", "Civil", "AI-DS"]
SPECIALIZATIONS = ["Software", "Data Science", "Networks", "Embedded", "Design", "Analytics"]


def generate_synthetic_dataset(path: Path | None = None, rows: int = 750) -> pd.DataFrame:
    """Generate a documented demonstration dataset for reproducible ML demos."""
    path = path or CONFIG.dataset_path
    rng = np.random.default_rng(CONFIG.random_state)
    branch = rng.choice(BRANCHES, rows, p=[0.24, 0.2, 0.16, 0.13, 0.1, 0.17])
    specialization = rng.choice(SPECIALIZATIONS, rows)
    gender = rng.choice(["Female", "Male", "Other"], rows, p=[0.44, 0.54, 0.02])

    branch_boost = pd.Series(branch).map(
        {"Computer": 7, "IT": 6, "AI-DS": 6, "ENTC": 2, "Mechanical": -1, "Civil": -2}
    ).to_numpy()
    cgpa = np.clip(rng.normal(7.2, 1.0, rows), 4.5, 10.0).round(2)
    attendance = np.clip(rng.normal(78, 10, rows), 45, 100).round(1)
    backlogs = np.clip(rng.poisson(np.maximum(0.4, 2.3 - cgpa / 4)), 0, 8)
    coding = np.clip(rng.normal(58 + branch_boost + cgpa * 2.1, 14, rows), 10, 100).round(1)
    dsa = np.clip(coding + rng.normal(-2, 11, rows), 5, 100).round(1)
    aptitude = np.clip(rng.normal(55 + cgpa * 3.0, 13, rows), 10, 100).round(1)
    communication = np.clip(rng.normal(62 + attendance * 0.12, 12, rows), 10, 100).round(1)
    interview = np.clip(0.35 * communication + 0.25 * aptitude + 0.25 * coding + rng.normal(8, 9, rows), 10, 100).round(1)
    projects = np.clip(rng.poisson(np.maximum(0.5, (coding - 35) / 22)), 0, 8)
    certifications = np.clip(rng.poisson(np.maximum(0.4, (coding + aptitude - 90) / 45)), 0, 8)
    internships = np.clip(rng.binomial(3, np.clip((cgpa + coding / 20 - 6) / 8, 0.05, 0.8)), 0, 3)
    hackathons = np.clip(rng.poisson(np.maximum(0.2, (coding - 45) / 30)), 0, 6)

    logit = (
        -11.4
        + 0.42 * cgpa
        + 0.018 * attendance
        - 0.36 * backlogs
        + 0.018 * coding
        + 0.015 * dsa
        + 0.017 * aptitude
        + 0.014 * communication
        + 0.022 * interview
        + 0.25 * projects
        + 0.22 * certifications
        + 0.56 * internships
        + 0.14 * hackathons
        + np.where(np.isin(branch, ["Computer", "IT", "AI-DS"]), 0.35, -0.1)
    )
    probability = 1 / (1 + np.exp(-logit))
    placed = rng.binomial(1, probability)
    package = np.where(
        placed == 1,
        np.clip(
            2.2 + cgpa * 0.45 + coding * 0.025 + projects * 0.25 + internships * 0.7 + rng.normal(0, 0.8, rows),
            2.0,
            18.0,
        ),
        0.0,
    ).round(2)

    df = pd.DataFrame(
        {
            "student_id": [f"STU{1000 + i}" for i in range(rows)],
            "student_name": [f"Student {i + 1}" for i in range(rows)],
            "gender": gender,
            "branch": branch,
            "specialization": specialization,
            "cgpa": cgpa,
            "attendance_pct": attendance,
            "backlogs": backlogs,
            "coding_score": coding,
            "dsa_score": dsa,
            "aptitude_score": aptitude,
            "communication_score": communication,
            "interview_score": interview,
            "projects": projects,
            "certifications": certifications,
            "internships": internships,
            "hackathons": hackathons,
            "placed": placed.astype(int),
            "package_lpa": package,
        }
    )
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


def load_dataset(path: Path | None = None, create_if_missing: bool = True) -> pd.DataFrame:
    path = path or CONFIG.dataset_path
    if not path.exists():
        if create_if_missing:
            LOGGER.info("Dataset missing; generating demonstration dataset at %s", path)
            return generate_synthetic_dataset(path)
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def validate_dataset(df: pd.DataFrame) -> dict[str, object]:
    required = set(CONFIG.numeric_features + CONFIG.categorical_features + (CONFIG.target,))
    missing = sorted(required - set(df.columns))
    duplicated = int(df.duplicated().sum())
    missing_values = df.isna().sum().sort_values(ascending=False)
    target_valid = CONFIG.target in df and set(df[CONFIG.target].dropna().unique()).issubset({0, 1})
    return {
        "is_valid": not missing and target_valid,
        "missing_required_columns": missing,
        "duplicate_rows": duplicated,
        "missing_values": missing_values[missing_values > 0].to_dict(),
        "target_is_binary": bool(target_valid),
        "shape": df.shape,
    }


def feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric = [col for col in CONFIG.numeric_features if col in df.columns]
    categorical = [col for col in CONFIG.categorical_features if col in df.columns]
    return numeric, categorical

