"""
ML Model loading & inference wrapper.

Loads the AI team's artifacts from:
  - modals/best_diabetes_model.pkl  (Logistic Regression — best model)
  - modals/scaler.pkl

Fallback paths (project root) are also supported:
  - diabetes_model.pkl / best_diabetes_model.pkl
  - scaler.pkl

NOTE: The selected Logistic Regression model was trained on *unscaled* features
in the AI notebook. Inference therefore uses raw feature values for LR/DT.
Scaler is still loaded for SVM/MLP-style models and for admin tooling.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
MODALS = ROOT / "modals"

FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# Official evaluation metrics from the AI team (details about models.docx)
OFFICIAL_METRICS = [
    ("Logistic Regression", 0.744589, 0.671875, 0.530864, 0.593103),
    ("Support Vector Machine (SVM)", 0.740260, 0.672131, 0.506173, 0.577465),
    ("Decision Tree", 0.692641, 0.565789, 0.530864, 0.547771),
    ("Neural Networks", 0.705628, 0.586667, 0.543210, 0.564103),
]

BEST_MODEL_NAME = "Logistic Regression"
BEST_MODEL_ACCURACY = 0.7446


def _resolve_model_path() -> Path | None:
    candidates = [
        MODALS / "best_diabetes_model.pkl",
        ROOT / "best_diabetes_model.pkl",
        ROOT / "diabetes_model.pkl",
        MODALS / "diabetes_model.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def _resolve_scaler_path() -> Path | None:
    candidates = [
        MODALS / "scaler.pkl",
        ROOT / "scaler.pkl",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def load_artifacts():
    """Loads the pre-trained ML model and standard scaler."""
    try:
        model_path = _resolve_model_path()
        scaler_path = _resolve_scaler_path()
        if model_path is None:
            return None, None
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path) if scaler_path is not None else None
        return model, scaler
    except Exception:
        return None, None


def _features_frame(features_list) -> pd.DataFrame:
    return pd.DataFrame([features_list], columns=FEATURE_NAMES)


def _needs_scaling(model) -> bool:
    """LR/DT in the AI notebook were fit on raw features; SVM/MLP used scaled."""
    name = type(model).__name__
    return name not in {"LogisticRegression", "DecisionTreeClassifier"}


def _model_matrix(model, scaler, features_list):
    frame = _features_frame(features_list)
    if scaler is not None and _needs_scaling(model):
        return scaler.transform(frame)
    return frame


def model_status() -> dict:
    """Expose artifact presence for Admin / debugging UI."""
    model, scaler = load_artifacts()
    model_path = _resolve_model_path()
    scaler_path = _resolve_scaler_path()
    return {
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "model_path": str(model_path) if model_path else None,
        "scaler_path": str(scaler_path) if scaler_path else None,
        "using_fallback": model is None,
        "best_model_name": BEST_MODEL_NAME,
        "best_model_accuracy": BEST_MODEL_ACCURACY,
        "model_type": type(model).__name__ if model is not None else None,
    }


def predict_risk(features_list):
    """
    Interface function for AI predictions.

    Parameters:
        features_list (list): [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]

    Returns:
        tuple: (prediction_class [0 or 1], confidence_percentage [float])
    """
    model, scaler = load_artifacts()

    if model is not None:
        matrix = _model_matrix(model, scaler, features_list)
        prediction = int(model.predict(matrix)[0])
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(matrix)[0]
            confidence = float(np.max(probabilities) * 100)
        else:
            confidence = 75.0
        return prediction, confidence

    # Fallback simulation if model files are missing during UI testing
    glucose = features_list[1]
    simulated_pred = 1 if glucose > 125 else 0
    return simulated_pred, 85.0


def explain_risk_factors(features_list) -> list[dict]:
    """
    Interpretable contribution ranking for the loaded linear model.

    Returns list of {feature, value, coefficient, contribution, direction}
    sorted by absolute contribution (highest impact first).
    """
    model, _ = load_artifacts()
    if model is None or not hasattr(model, "coef_"):
        # Heuristic ranking when model coeffs unavailable
        names = FEATURE_NAMES
        heuristic = [0.05, 0.35, 0.05, 0.05, 0.1, 0.25, 0.1, 0.05]
        rows = []
        for name, value, weight in zip(names, features_list, heuristic):
            rows.append(
                {
                    "feature": name,
                    "value": float(value),
                    "coefficient": float(weight),
                    "contribution": float(value) * float(weight),
                    "direction": "increases risk" if weight >= 0 else "decreases risk",
                }
            )
        return sorted(rows, key=lambda r: abs(r["contribution"]), reverse=True)

    coefs = model.coef_[0]
    rows = []
    for name, value, coef in zip(FEATURE_NAMES, features_list, coefs):
        contribution = float(coef) * float(value)
        rows.append(
            {
                "feature": name,
                "value": float(value),
                "coefficient": float(coef),
                "contribution": contribution,
                "direction": "increases risk" if contribution >= 0 else "decreases risk",
            }
        )
    return sorted(rows, key=lambda r: abs(r["contribution"]), reverse=True)


def feature_importance_table() -> pd.DataFrame:
    """Global feature importance from absolute logistic coefficients."""
    model, _ = load_artifacts()
    if model is None or not hasattr(model, "coef_"):
        return pd.DataFrame(columns=["Feature", "Coefficient", "Abs Weight", "Effect"])
    coefs = model.coef_[0]
    rows = []
    for name, coef in zip(FEATURE_NAMES, coefs):
        rows.append(
            {
                "Feature": name,
                "Coefficient": float(coef),
                "Abs Weight": abs(float(coef)),
                "Effect": "Raises risk" if coef > 0 else "Lowers risk",
            }
        )
    return pd.DataFrame(rows).sort_values("Abs Weight", ascending=False).reset_index(drop=True)


def evaluate_confusion_on_dataset(df: pd.DataFrame) -> dict | None:
    """
    Build a confusion matrix for the loaded best model on a labeled dataframe.
    Expects FEATURE_NAMES + Outcome columns.
    """
    from sklearn.metrics import confusion_matrix, classification_report

    model, scaler = load_artifacts()
    if model is None or "Outcome" not in df.columns:
        return None

    work = df.copy()
    missing = [c for c in FEATURE_NAMES if c not in work.columns]
    if missing:
        return None

    X = work[FEATURE_NAMES]
    y = work["Outcome"].astype(int)
    if _needs_scaling(model) and scaler is not None:
        preds = model.predict(scaler.transform(X))
    else:
        preds = model.predict(X)

    cm = confusion_matrix(y, preds, labels=[0, 1])
    report = classification_report(y, preds, output_dict=True, zero_division=0)
    return {
        "matrix": cm,
        "tn": int(cm[0, 0]),
        "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]),
        "tp": int(cm[1, 1]),
        "report": report,
    }
