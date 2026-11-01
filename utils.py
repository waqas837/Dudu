"""Data cleaning, normalization, education content, and utility helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

ZERO_IMPUTE_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

FEATURE_COLUMNS = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

TARGET_COLUMN = "Outcome"

DATA_DIR = Path(__file__).resolve().parent / "data"
TRAINING_CSV_PATH = DATA_DIR / "training_data.csv"

EDUCATION_SECTIONS = [
    {
        "title": "What is diabetes risk?",
        "body": (
            "Diabetes risk means how likely someone is to develop type 2 diabetes "
            "based on clinical markers such as glucose, BMI, age, family history "
            "(diabetes pedigree), blood pressure, and insulin response."
        ),
    },
    {
        "title": "Key risk factors",
        "body": (
            "Higher fasting glucose, higher BMI, older age, and stronger family "
            "history generally raise risk. Skin thickness and insulin values help "
            "capture metabolic patterns used by the prediction model."
        ),
    },
    {
        "title": "Prevention strategies",
        "body": (
            "Aim for 150+ minutes of moderate activity per week, reduce sugary drinks, "
            "prefer whole grains and vegetables, maintain a healthy weight, sleep 7–8 hours, "
            "and avoid smoking."
        ),
    },
    {
        "title": "Healthy lifestyle guidelines",
        "body": (
            "Track glucose and weight over time, schedule routine checkups, manage stress, "
            "and follow clinician advice for medication or further labs. Early lifestyle "
            "changes can meaningfully lower long-term risk."
        ),
    },
    {
        "title": "Important disclaimer",
        "body": (
            "This application is an educational / clinical decision-support tool. "
            "It does not diagnose diabetes and must not replace professional medical care."
        ),
    },
]


def ensure_data_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def impute_invalid_zeros(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace medically invalid zeros with column medians.

    Applies to Glucose, BloodPressure, SkinThickness, Insulin, BMI.
    """
    cleaned = df.copy()
    alias_map = {
        "glucose": "Glucose",
        "bloodpressure": "BloodPressure",
        "blood_pressure": "BloodPressure",
        "skinthickness": "SkinThickness",
        "skin_thickness": "SkinThickness",
        "insulin": "Insulin",
        "bmi": "BMI",
    }

    for col in cleaned.columns:
        key = col.replace(" ", "").lower()
        if key in alias_map or col in ZERO_IMPUTE_COLUMNS:
            series = pd.to_numeric(cleaned[col], errors="coerce")
            median = series[series != 0].median()
            if pd.isna(median):
                median = 0.0
            series = series.replace(0, median)
            cleaned[col] = series.fillna(median)
    return cleaned


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map common CSV headers to the canonical Pima diabetes schema."""
    renamed = df.copy()
    mapping = {}
    for col in renamed.columns:
        key = col.strip().lower().replace(" ", "").replace("_", "")
        if key in ("pregnancies", "preg"):
            mapping[col] = "Pregnancies"
        elif key == "glucose":
            mapping[col] = "Glucose"
        elif key in ("bloodpressure", "bp"):
            mapping[col] = "BloodPressure"
        elif key in ("skinthickness", "skin"):
            mapping[col] = "SkinThickness"
        elif key == "insulin":
            mapping[col] = "Insulin"
        elif key == "bmi":
            mapping[col] = "BMI"
        elif key in ("diabetespedigreefunction", "dpf", "diabetespedigree"):
            mapping[col] = "DiabetesPedigreeFunction"
        elif key == "age":
            mapping[col] = "Age"
        elif key in ("outcome", "class", "label"):
            mapping[col] = "Outcome"
    return renamed.rename(columns=mapping)


def clean_uploaded_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Normalize headers, impute invalid zeros, validate required columns."""
    normalized = normalize_column_names(df)
    missing = [c for c in FEATURE_COLUMNS if c not in normalized.columns]
    if missing:
        return normalized, f"Missing required columns: {', '.join(missing)}"
    cleaned = impute_invalid_zeros(normalized)
    return cleaned, "Dataset cleaned successfully (invalid zeros replaced with medians)."


def save_training_dataset(df: pd.DataFrame) -> Path:
    ensure_data_dir()
    df.to_csv(TRAINING_CSV_PATH, index=False)
    return TRAINING_CSV_PATH


def load_training_dataset() -> pd.DataFrame | None:
    if not TRAINING_CSV_PATH.exists():
        return None
    return pd.read_csv(TRAINING_CSV_PATH)


def personalized_advice(glucose: float, bmi: float) -> list[str]:
    """Rule-based recommendations from Glucose & BMI thresholds."""
    tips: list[str] = []

    if glucose >= 126:
        tips.append(
            "Your fasting glucose is in the diabetic range (≥126 mg/dL). "
            "Schedule a clinician visit promptly and discuss confirmatory testing."
        )
    elif glucose >= 100:
        tips.append(
            "Your glucose is in the prediabetes range (100–125 mg/dL). "
            "Focus on reducing refined carbs and recheck levels regularly."
        )
    else:
        tips.append("Glucose looks within a typical fasting range. Keep monitoring periodically.")

    if bmi >= 30:
        tips.append(
            "BMI is in the obese range (≥30). Aim for gradual weight loss with "
            "a calorie-controlled diet and at least 150 minutes of activity per week."
        )
    elif bmi >= 25:
        tips.append(
            "BMI is in the overweight range (25–29.9). Increase daily walking and "
            "prioritize whole foods over ultra-processed meals."
        )
    else:
        tips.append("BMI is in a healthy range. Maintain balanced nutrition and regular exercise.")

    tips.append(
        "General tips: stay hydrated, limit sugary drinks, get 7–8 hours of sleep, "
        "and never stop or start medication without medical advice."
    )
    tips.append(
        "Disclaimer: this advice is educational only and does not replace professional medical care."
    )
    return tips
