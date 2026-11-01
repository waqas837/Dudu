"""Patient form, advice, explanations, longitudinal charts, feedback, reports."""

from __future__ import annotations

import streamlit as st
import plotly.express as px
import pandas as pd

import db_helper
import ml_engine
import report_helper
from utils import EDUCATION_SECTIONS, personalized_advice

st.set_page_config(page_title="Patient Portal", page_icon="🧑‍⚕️", layout="wide")
db_helper.init_db()


def _ensure_session_defaults() -> None:
    for key, value in {
        "authenticated": False,
        "user_id": None,
        "username": None,
        "role": None,
        "email": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value


def require_patient() -> bool:
    _ensure_session_defaults()
    if not st.session_state.get("authenticated"):
        st.warning("Not signed in. Use a demo login below or open the Home page.")
        c1, c2 = st.columns(2)
        if c1.button("Patient demo login", key="portal_demo_patient"):
            user = db_helper.authenticate_user("patient1", "patient123")
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user["user_id"]
                st.session_state.username = user["username"]
                st.session_state.role = user["role"]
                st.session_state.email = user["email"]
                st.rerun()
        if c2.button("Go to Home", key="portal_go_home"):
            st.switch_page("app.py")
        st.stop()
        return False
    if st.session_state.get("role") != "Patient":
        st.error("This portal is only available to Patient accounts.")
        st.stop()
        return False
    return True


def render_prediction_form() -> None:
    st.subheader("Clinical Feature Inputs")
    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0, step=1)
        glucose = st.slider("Glucose (mg/dL)", 0, 200, 110)
        blood_pressure = st.slider("Blood Pressure (mm Hg)", 0, 150, 70)
        skin_thickness = st.slider("Skin Thickness (mm)", 0, 100, 20)
    with c2:
        insulin = st.slider("Insulin (µU/ml)", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 28.0, step=0.1)
        diabetes_pedigree = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.00,
            max_value=3.00,
            value=0.50,
            step=0.01,
            format="%.2f",
        )
        age = st.slider("Age (years)", 1, 100, 35)

    if not st.button("Predict Risk", width="stretch", type="primary", key="predict_btn"):
        return

    features = [
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        diabetes_pedigree,
        age,
    ]
    prediction, confidence = ml_engine.predict_risk(features)
    db_helper.insert_patient_record(
        patient_id=st.session_state.user_id,
        pregnancies=pregnancies,
        glucose=glucose,
        blood_pressure=blood_pressure,
        skin_thickness=skin_thickness,
        insulin=insulin,
        bmi=bmi,
        diabetes_pedigree=diabetes_pedigree,
        age=age,
        prediction_result=int(prediction),
        risk_probability=float(confidence),
    )

    status = ml_engine.model_status()
    if status["using_fallback"]:
        st.caption("Using fallback risk rule (ML artifacts not loaded yet).")
    else:
        st.caption(
            f"Predicted with **{status['best_model_name']}** "
            f"(accuracy {status['best_model_accuracy'] * 100:.2f}%)."
        )

    st.markdown("### Risk Assessment")
    if prediction == 1:
        st.error(f"**High Risk of Diabetes** · confidence {confidence:.1f}%")
    else:
        st.success(f"**Low Risk / Normal** · confidence {confidence:.1f}%")

    explanations = ml_engine.explain_risk_factors(features)
    st.markdown("### Why this prediction? (Risk factor analysis)")
    exp_df = pd.DataFrame(explanations)[["feature", "value", "contribution", "direction"]]
    exp_df = exp_df.rename(
        columns={
            "feature": "Feature",
            "value": "Your Value",
            "contribution": "Impact Score",
            "direction": "Effect",
        }
    )
    st.dataframe(exp_df, width="stretch", hide_index=True)
    top = explanations[0]
    st.info(
        f"Strongest driver for this assessment: **{top['feature']}** "
        f"(value {top['value']:.2f}) — {top['direction']}."
    )

    advice = personalized_advice(glucose, bmi)
    st.markdown("### Personalized Recommendations")
    for tip in advice:
        st.write(f"- {tip}")

    st.session_state["last_prediction_bundle"] = {
        "prediction": prediction,
        "confidence": confidence,
        "advice": advice,
        "explanations": explanations,
    }


def render_history() -> None:
    st.subheader("Your Health History")
    df = db_helper.get_patient_records(st.session_state.user_id)
    if df.empty:
        st.info("No records yet. Submit a prediction to start tracking.")
        return

    display = df.copy()
    display["risk_label"] = display["prediction_result"].map({0: "Low Risk", 1: "High Risk"})
    st.dataframe(display, width="stretch", hide_index=True)

    chart_df = df.copy()
    chart_df["recorded_at"] = chart_df["recorded_at"].astype(str)
    long_df = chart_df.melt(
        id_vars=["recorded_at"],
        value_vars=["glucose", "bmi"],
        var_name="metric",
        value_name="value",
    )
    fig = px.line(
        long_df,
        x="recorded_at",
        y="value",
        color="metric",
        markers=True,
        title="Glucose & BMI vs Time",
    )
    fig.update_layout(xaxis_title="Recorded At", yaxis_title="Value")
    st.plotly_chart(fig, width="stretch")

    csv_bytes = display.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV Report",
        data=csv_bytes,
        file_name=f"patient_{st.session_state.username}_history.csv",
        mime="text/csv",
        width="stretch",
        key="csv_report_btn",
    )

    bundle = st.session_state.get("last_prediction_bundle") or {}
    last = df.iloc[-1]
    latest = {
        "prediction": int(last["prediction_result"]),
        "confidence": float(last["risk_probability"] or 0),
    }
    if bundle:
        latest = {
            "prediction": int(bundle.get("prediction", latest["prediction"])),
            "confidence": float(bundle.get("confidence", latest["confidence"])),
        }

    try:
        pdf_bytes = report_helper.build_patient_pdf(
            username=st.session_state.username,
            history_df=display,
            latest_prediction=latest,
            advice=bundle.get("advice"),
            explanations=bundle.get("explanations"),
        )
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name=f"patient_{st.session_state.username}_report.pdf",
            mime="application/pdf",
            width="stretch",
            key="pdf_report_btn",
        )
    except Exception as exc:
        st.warning(f"PDF report unavailable right now: {exc}")


def render_feedback() -> None:
    st.subheader("Feedback")
    rating = st.slider("Rating", 1, 5, 5, key="fb_rating")
    comments = st.text_area(
        "Comments",
        placeholder="Tell us how useful the assessment was.",
        key="fb_comments",
    )
    if st.button("Submit Feedback", width="stretch", key="fb_submit"):
        if not comments.strip():
            st.warning("Please add a short comment.")
        else:
            db_helper.insert_feedback(
                st.session_state.user_id,
                st.session_state.role,
                rating,
                comments,
            )
            st.success("Thank you — feedback saved.")


def render_education() -> None:
    st.subheader("Education & Resources")
    st.caption("Diabetes risk factors, prevention, and healthy lifestyle guidance.")
    for section in EDUCATION_SECTIONS:
        with st.expander(section["title"], expanded=False):
            st.write(section["body"])


def main() -> None:
    require_patient()
    st.title("Patient Portal")
    st.caption(f"Logged in as {st.session_state.username}")

    tab_predict, tab_history, tab_feedback, tab_edu = st.tabs(
        ["Predict Risk", "History & Trends", "Feedback", "Education"]
    )
    with tab_predict:
        render_prediction_form()
    with tab_history:
        render_history()
    with tab_feedback:
        render_feedback()
    with tab_edu:
        render_education()


main()
