from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
MODEL_DIR = ROOT_DIR / "models"


@dataclass(frozen=True)
class ProjectConfig:
    random_state: int = 42
    test_size: float = 0.2
    dataset_path: Path = DATA_RAW_DIR / "student_placement_demo.csv"
    target: str = "placed"
    regression_target: str = "package_lpa"
    id_column: str = "student_id"
    excluded_from_model: tuple[str, ...] = (
        "student_id",
        "student_name",
        "gender",
        "placed",
        "package_lpa",
    )
    sensitive_columns: tuple[str, ...] = ("gender",)
    positive_label: int = 1
    model_metric: str = "f1"
    numeric_features: tuple[str, ...] = (
        "cgpa",
        "attendance_pct",
        "backlogs",
        "coding_score",
        "dsa_score",
        "aptitude_score",
        "communication_score",
        "interview_score",
        "projects",
        "certifications",
        "internships",
        "hackathons",
    )
    categorical_features: tuple[str, ...] = ("branch", "specialization")
    readiness_groups: dict[str, tuple[str, ...]] = field(
        default_factory=lambda: {
            "Academic Performance": ("cgpa", "attendance_pct", "backlogs"),
            "Technical Skills": ("coding_score", "dsa_score"),
            "Aptitude": ("aptitude_score",),
            "Communication": ("communication_score", "interview_score"),
            "Experience": ("internships", "certifications", "hackathons"),
            "Projects": ("projects",),
        }
    )
    feature_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "cgpa": 7.0,
            "attendance_pct": 75.0,
            "backlogs": 1.0,
            "coding_score": 65.0,
            "dsa_score": 60.0,
            "aptitude_score": 65.0,
            "communication_score": 65.0,
            "interview_score": 65.0,
            "projects": 2.0,
            "certifications": 2.0,
            "internships": 1.0,
            "hackathons": 1.0,
        }
    )


CONFIG = ProjectConfig()

