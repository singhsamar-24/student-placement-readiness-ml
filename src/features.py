from __future__ import annotations

import pandas as pd


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add transparent, leakage-free derived features."""
    out = df.copy()
    if {"coding_score", "dsa_score"}.issubset(out.columns):
        out["technical_index"] = out[["coding_score", "dsa_score"]].mean(axis=1)
    if {"aptitude_score", "communication_score", "interview_score"}.issubset(out.columns):
        out["interview_readiness_index"] = out[["aptitude_score", "communication_score", "interview_score"]].mean(axis=1)
    if {"projects", "certifications", "internships", "hackathons"}.issubset(out.columns):
        out["portfolio_strength"] = (
            out["projects"] * 1.6 + out["certifications"] + out["internships"] * 2.2 + out["hackathons"] * 0.8
        )
    if {"cgpa", "backlogs"}.issubset(out.columns):
        out["academic_stability"] = out["cgpa"] * 10 - out["backlogs"] * 6
    return out


def engineered_numeric_features(existing_columns: list[str]) -> list[str]:
    candidates = [
        "technical_index",
        "interview_readiness_index",
        "portfolio_strength",
        "academic_stability",
    ]
    return [col for col in candidates if col in existing_columns]

