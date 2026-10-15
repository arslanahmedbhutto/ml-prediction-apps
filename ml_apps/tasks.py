"""The three regression tasks: data, features, target and input ranges for the app."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
SEED = 42


@dataclass
class Task:
    key: str
    title: str
    file: str
    target: str
    target_label: str
    features: list[str]
    # feature -> (label, min, max, default, step) used by the app's inputs
    inputs: dict[str, tuple] = field(default_factory=dict)
    # Leading rows that come from a different source and are excluded (see README)
    skip_first: int = 0

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(DATA_DIR / self.file).iloc[self.skip_first:]
        return df[self.features + [self.target]].dropna().reset_index(drop=True)


GPA = Task(
    key="gpa", title="Student GPA", file="student_gpa.csv", target="GPA", target_label="GPA (0–4)",
    features=["Age", "Gender", "Ethnicity", "ParentalEducation", "StudyTimeWeekly", "Absences", "Tutoring",
              "ParentalSupport", "Extracurricular", "Sports", "Music", "Volunteering"],
    inputs={
        "Age": ("Age", 15, 18, 16, 1),
        "Gender": ("Gender (0 = male, 1 = female)", 0, 1, 0, 1),
        "Ethnicity": ("Ethnicity group (0–3)", 0, 3, 0, 1),
        "ParentalEducation": ("Parental education (0 none – 4 higher)", 0, 4, 2, 1),
        "StudyTimeWeekly": ("Study time per week (hours)", 0.0, 20.0, 10.0, 0.5),
        "Absences": ("Absences in the year", 0, 30, 5, 1),
        "Tutoring": ("Tutoring (0/1)", 0, 1, 0, 1),
        "ParentalSupport": ("Parental support (0 none – 4 very high)", 0, 4, 2, 1),
        "Extracurricular": ("Extracurricular (0/1)", 0, 1, 0, 1),
        "Sports": ("Sports (0/1)", 0, 1, 0, 1),
        "Music": ("Music (0/1)", 0, 1, 0, 1),
        "Volunteering": ("Volunteering (0/1)", 0, 1, 0, 1),
    },
)

HOUSE = Task(
    key="house", title="House price", file="house_prices.csv", target="Price", target_label="Price",
    features=["Area", "Bedrooms", "Age"],
    inputs={
        "Area": ("Area (sq ft)", 2000, 4000, 3000, 50),
        "Bedrooms": ("Bedrooms", 3, 10, 6, 1),
        "Age": ("Age (years)", 5, 20, 12, 1),
    },
    # Rows 0-10 are an 11-row hand-made teaching set (with one bedroom value imputed as 5.5) pasted on
    # top of 489 generated rows. They follow a different price rule and caused all of the large errors.
    skip_first=11,
)

SALARY = Task(
    key="salary", title="Salary", file="salary_experience.csv", target="Salary", target_label="Salary",
    features=["YearsExperience"],
    inputs={"YearsExperience": ("Years of experience", 0.0, 15.0, 5.0, 0.5)},
)

TASKS = {t.key: t for t in (GPA, HOUSE, SALARY)}
