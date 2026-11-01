"""End-to-end backend smoke test. Run: python smoke_test.py"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

import db_helper
import ml_engine
import report_helper
from utils import clean_uploaded_dataset, personalized_advice, save_training_dataset


def main() -> int:
    db_helper.init_db()
    db_helper.sync_official_metrics()
    failures: list[str] = []

    status = ml_engine.model_status()
    if status["using_fallback"]:
        failures.append("model should be loaded from modals/")
    if status.get("best_model_name") != "Logistic Regression":
        failures.append("best model name mismatch")

    # Auth
    patient = db_helper.authenticate_user("patient1", "patient123")
    doctor = db_helper.authenticate_user("doctor1", "doctor123")
    admin = db_helper.authenticate_user("admin1", "admin123")
    if not patient or patient["role"] != "Patient":
        failures.append("patient auth failed")
    if not doctor or doctor["role"] != "Doctor":
        failures.append("doctor auth failed")
    if not admin or admin["role"] != "Admin":
        failures.append("admin auth failed")

    # High-risk profile should classify as 1 with unscaled LR
    pred, conf = ml_engine.predict_risk([6, 160, 80, 30, 150, 35.0, 0.8, 50])
    if pred != 1:
        failures.append(f"expected high risk for elevated profile, got {pred}")
    if conf < 50:
        failures.append(f"confidence unexpectedly low: {conf}")

    low_pred, _ = ml_engine.predict_risk([0, 90, 70, 20, 50, 22.0, 0.2, 25])
    if low_pred != 0:
        failures.append(f"expected low risk for healthy profile, got {low_pred}")

    rid = db_helper.insert_patient_record(
        patient_id=patient["user_id"],
        pregnancies=6,
        glucose=160,
        blood_pressure=80,
        skin_thickness=30,
        insulin=150,
        bmi=35.0,
        diabetes_pedigree=0.8,
        age=50,
        prediction_result=pred,
        risk_probability=conf,
    )
    if rid < 1:
        failures.append("insert_patient_record failed")

    explanations = ml_engine.explain_risk_factors([6, 160, 80, 30, 150, 35.0, 0.8, 50])
    if len(explanations) < 5:
        failures.append("explain_risk_factors too short")

    tips = personalized_advice(160, 35.0)
    if len(tips) < 3:
        failures.append("advice too short")

    hist = db_helper.get_patient_records(patient["user_id"])
    pdf = report_helper.build_patient_pdf(
        username=patient["username"],
        history_df=hist,
        latest_prediction={"prediction": pred, "confidence": conf},
        advice=tips,
        explanations=explanations,
    )
    if not pdf.startswith(b"%PDF"):
        failures.append("PDF report invalid")

    db_helper.insert_feedback(patient["user_id"], "Patient", 5, "Full build smoke feedback")
    metrics = db_helper.get_model_metrics()
    if metrics.empty:
        failures.append("model metrics empty")
    else:
        lr = metrics[metrics["model_name"] == "Logistic Regression"]
        if lr.empty or float(lr.iloc[0]["accuracy"]) < 0.74:
            failures.append("official LR metric missing/wrong")

    sample = Path("data/sample_diabetes.csv")
    if sample.exists():
        raw = pd.read_csv(sample)
        cleaned, msg = clean_uploaded_dataset(raw)
        if "Missing" in msg:
            failures.append(msg)
        else:
            save_training_dataset(cleaned)
            cm = ml_engine.evaluate_confusion_on_dataset(cleaned)
            if cm is None:
                failures.append("confusion matrix failed")

    if failures:
        print("FAIL:")
        for f in failures:
            print(" -", f)
        return 1

    print("PASS: model wired, auth, prediction, explanations, PDF, metrics, CM OK")
    print(f"  model={status['model_type']} path={status['model_path']}")
    print(f"  high_risk_pred={pred} conf={conf:.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
