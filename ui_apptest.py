"""Streamlit AppTest coverage. Run: python ui_apptest.py"""

from __future__ import annotations

import sys

from streamlit.testing.v1 import AppTest

import db_helper
import ml_engine


def login(at: AppTest, username: str, password: str) -> AppTest:
    at.text_input(key="login_username").set_value(username)
    at.text_input(key="login_password").set_value(password)
    at.button(key="login_btn").click().run()
    return at


def main() -> int:
    db_helper.init_db()
    db_helper.sync_official_metrics()
    failures: list[str] = []

    if ml_engine.model_status()["using_fallback"]:
        failures.append("model artifacts not loaded")

    at = AppTest.from_file("app.py", default_timeout=30).run()
    login(at, "patient1", "patient123")
    if not at.session_state.get("authenticated"):
        failures.append("patient login failed")

    patient = AppTest.from_file("pages/1_Patient_Portal.py", default_timeout=60)
    patient.session_state["authenticated"] = True
    patient.session_state["user_id"] = 1
    patient.session_state["username"] = "patient1"
    patient.session_state["role"] = "Patient"
    patient.session_state["email"] = "patient1@demo.com"
    patient.run()

    # Raise glucose toward high risk and predict
    for s in patient.slider:
        if "Glucose" in str(s.label):
            s.set_value(160)
            break
    patient.button(key="predict_btn").click().run()

    joined = " ".join([*(e.value for e in patient.error), *(s.value for s in patient.success)])
    joined += " " + " ".join(t.value for t in patient.markdown)
    if "High Risk" not in joined and "Low Risk" not in joined:
        failures.append(f"prediction panel missing: {joined[:250]}")

    hist = db_helper.get_patient_records(1)
    if hist.empty:
        failures.append("patient record not written")

    # Fresh run for feedback widgets (predict path may leave tab-local state)
    patient2 = AppTest.from_file("pages/1_Patient_Portal.py", default_timeout=45)
    patient2.session_state["authenticated"] = True
    patient2.session_state["user_id"] = 1
    patient2.session_state["username"] = "patient1"
    patient2.session_state["role"] = "Patient"
    patient2.session_state["email"] = "patient1@demo.com"
    patient2.run()
    patient2.text_area(key="fb_comments").set_value("UI apptest feedback full build")
    patient2.button(key="fb_submit").click().run()

    doctor = AppTest.from_file("pages/2_Doctor_Portal.py", default_timeout=60)
    doctor.session_state["authenticated"] = True
    doctor.session_state["user_id"] = 2
    doctor.session_state["username"] = "doctor1"
    doctor.session_state["role"] = "Doctor"
    doctor.session_state["email"] = "doctor1@demo.com"
    doctor.run()
    if doctor.error:
        failures.append(f"doctor portal errors: {[e.value for e in doctor.error]}")

    admin = AppTest.from_file("pages/3_Admin_Portal.py", default_timeout=60)
    admin.session_state["authenticated"] = True
    admin.session_state["user_id"] = 3
    admin.session_state["username"] = "admin1"
    admin.session_state["role"] = "Admin"
    admin.session_state["email"] = "admin1@demo.com"
    admin.run()
    titles = [t.value for t in admin.title]
    if not any("Administrator" in t for t in titles):
        failures.append(f"admin title missing: {titles}")

    if failures:
        print("FAIL")
        for f in failures:
            print(" -", f)
        return 1

    print("PASS: AppTest login + patient predict + doctor + admin with live model")
    return 0


if __name__ == "__main__":
    sys.exit(main())
