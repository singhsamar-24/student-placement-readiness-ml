from __future__ import annotations

import pandas as pd

from src.config import CONFIG


MESSAGES = {
    "cgpa": "Strengthen academic consistency; prioritize core subjects and raise CGPA above the placement benchmark.",
    "attendance_pct": "Improve attendance and classroom participation to reduce academic risk signals.",
    "backlogs": "Resolve active or past backlog issues and prepare a clear academic improvement story.",
    "coding_score": "Practice DSA and coding problems regularly; build timed problem-solving discipline.",
    "dsa_score": "Focus on arrays, strings, trees, graphs, dynamic programming, and company-style DSA rounds.",
    "aptitude_score": "Work on quantitative aptitude, logical reasoning, and timed assessment practice.",
    "communication_score": "Schedule mock interviews and practice concise project explanations.",
    "interview_score": "Run technical and HR mock interviews; improve examples using STAR-style answers.",
    "projects": "Build 2-3 relevant projects with clean GitHub documentation and measurable outcomes.",
    "certifications": "Complete targeted certifications aligned with your preferred role.",
    "internships": "Apply for internships, live projects, or open-source work to add industry exposure.",
    "hackathons": "Join hackathons or coding events to demonstrate initiative and teamwork.",
}


def weak_features(row: pd.Series | dict, limit: int = 5) -> list[str]:
    series = pd.Series(row)
    weak: list[tuple[str, float]] = []
    for feature, threshold in CONFIG.feature_thresholds.items():
        if feature not in series or pd.isna(series[feature]):
            continue
        value = float(series[feature])
        if feature == "backlogs":
            severity = value - threshold
            if value > threshold:
                weak.append((feature, severity))
        elif value < threshold:
            weak.append((feature, threshold - value))
    return [name for name, _ in sorted(weak, key=lambda item: item[1], reverse=True)[:limit]]


def generate_recommendations(row: pd.Series | dict, limit: int = 5) -> list[str]:
    features = weak_features(row, limit)
    recommendations = [MESSAGES[name] for name in features if name in MESSAGES]
    if not recommendations:
        recommendations.append("Maintain current momentum and keep improving through interview practice and portfolio updates.")
    return recommendations

