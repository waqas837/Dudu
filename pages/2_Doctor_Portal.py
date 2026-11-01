"""EDA dashboards & clinical decision support for clinicians."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import db_helper
import ml_engine

st.set_page_config(page_title="Doctor Portal", page_icon="👨‍⚕️", layout="wide")
db_helper.init_db()

DISCLAIMER = (
    "System designed as a clinical decision-support aid. "
    "Clinical judgment takes precedence."
)


def require_doctor() -> None:
    for key, value in {
        "authenticated": False,
        "user_id": None,
        "username": None,
        "role": None,
        "email": None,
    }.items():
        if key not in st.session_state:
            st.session_state[key] = value

    if not st.session_state.get("authenticated"):
        st.warning("Not signed in. Use a demo login below or open the Home page.")
        if st.button("Doctor demo login", key="portal_demo_doctor"):
            user = db_helper.authenticate_user("doctor1", "doctor123")
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user["user_id"]
                st.session_state.username = user["username"]
                st.session_state.role = user["role"]
                st.session_state.email = user["email"]
                st.rerun()
        st.stop()
    if st.session_state.get("role") not in ("Doctor", "Admin"):
        st.error("This portal is only available to Doctor (or Admin) accounts.")
        st.stop()


def render_kpis(df: pd.DataFrame) -> None:
    stats = db_helper.get_kpi_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Evaluated Patients", stats["total_evaluated"])
    c2.metric("Mean Glucose", f"{stats['mean_glucose']:.1f}")
    c3.metric("Mean BMI", f"{stats['mean_bmi']:.1f}")
    status = ml_engine.model_status()
    c4.metric(
        "Active Model",
        status["best_model_name"] if not status["using_fallback"] else "Fallback",
    )


def render_eda(df: pd.DataFrame) -> None:
    st.subheader("Clinical EDA Dashboard")
    if df.empty:
        st.info("No patient records available yet for EDA.")
        return

    plot_df = df.copy()
    plot_df["outcome_label"] = plot_df["prediction_result"].map(
        {0: "Low Risk", 1: "High Risk"}
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig, ax = plt.subplots(figsize=(6, 4))
        for label, subset in plot_df.groupby("outcome_label"):
            ax.hist(subset["glucose"], bins=12, alpha=0.55, label=label)
        ax.set_title("Glucose Distribution by Outcome")
        ax.set_xlabel("Glucose")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    with col_b:
        fig, ax = plt.subplots(figsize=(6, 4))
        melted = plot_df.melt(
            value_vars=["bmi", "insulin"],
            var_name="feature",
            value_name="value",
        )
        sns.boxplot(data=melted, x="feature", y="value", ax=ax)
        ax.set_title("BMI & Insulin Outlier View")
        st.pyplot(fig)
        plt.close(fig)

    feature_cols = [
        "pregnancies",
        "glucose",
        "blood_pressure",
        "skin_thickness",
        "insulin",
        "bmi",
        "diabetes_pedigree",
        "age",
        "prediction_result",
    ]
    available = [c for c in feature_cols if c in plot_df.columns]
    if len(available) >= 2:
        corr = plot_df[available].corr(numeric_only=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="Blues", ax=ax)
        ax.set_title("Feature Correlation Heatmap")
        st.pyplot(fig)
        plt.close(fig)


def render_cdss(df: pd.DataFrame) -> None:
    st.subheader("Clinical Decision Support")
    st.warning(DISCLAIMER)

    patients = db_helper.get_patient_list()
    if patients.empty:
        st.info("No patient accounts found.")
        return

    options = {
        f"{row.username} ({row.email}) — {int(row.record_count)} records": int(row.user_id)
        for _, row in patients.iterrows()
    }
    selected_label = st.selectbox("Select patient profile", list(options.keys()))
    patient_id = options[selected_label]
    patient_df = df[df["patient_id"] == patient_id].copy() if not df.empty else pd.DataFrame()

    if patient_df.empty:
        st.info("This patient has no recorded assessments yet.")
        return

    show = patient_df[
        [
            "recorded_at",
            "pregnancies",
            "glucose",
            "blood_pressure",
            "skin_thickness",
            "insulin",
            "bmi",
            "diabetes_pedigree",
            "age",
            "prediction_result",
            "risk_probability",
        ]
    ].copy()
    show["risk_label"] = show["prediction_result"].map({0: "Low Risk", 1: "High Risk"})
    st.dataframe(show, width="stretch", hide_index=True)

    latest = patient_df.sort_values("recorded_at").iloc[-1]
    features = [
        latest["pregnancies"],
        latest["glucose"],
        latest["blood_pressure"],
        latest["skin_thickness"],
        latest["insulin"],
        latest["bmi"],
        latest["diabetes_pedigree"],
        latest["age"],
    ]
    st.markdown("#### Risk-factor explanation for latest assessment")
    explanations = ml_engine.explain_risk_factors(features)
    exp_df = pd.DataFrame(explanations)[["feature", "value", "contribution", "direction"]]
    st.dataframe(exp_df, width="stretch", hide_index=True)


def render_model_insights() -> None:
    st.subheader("Model Insights")
    status = ml_engine.model_status()
    if status["using_fallback"]:
        st.warning("ML model artifacts are not loaded.")
        return

    st.success(
        f"Active model: **{status['best_model_name']}** "
        f"({status['model_type']}) · accuracy {status['best_model_accuracy'] * 100:.2f}%"
    )
    st.markdown("Global feature importance (from model coefficients)")
    st.dataframe(ml_engine.feature_importance_table(), width="stretch", hide_index=True)

    metrics = db_helper.get_model_metrics()
    if not metrics.empty:
        st.markdown("Official model comparison metrics")
        view = metrics.copy()
        for col in ("accuracy", "precision_score", "recall_score", "f1_score"):
            view[col] = (view[col] * 100).round(2)
        st.dataframe(
            view.rename(
                columns={
                    "model_name": "Model",
                    "accuracy": "Accuracy %",
                    "precision_score": "Precision %",
                    "recall_score": "Recall %",
                    "f1_score": "F1 %",
                }
            )[["Model", "Accuracy %", "Precision %", "Recall %", "F1 %"]],
            width="stretch",
            hide_index=True,
        )


def main() -> None:
    require_doctor()
    st.title("Doctor / Clinician Portal")
    st.caption(f"Logged in as {st.session_state.username} ({st.session_state.role})")

    df = db_helper.get_all_patient_records()
    render_kpis(df)

    tab_eda, tab_cdss, tab_model = st.tabs(
        ["EDA Dashboard", "Patient Decision Support", "Model Insights"]
    )
    with tab_eda:
        render_eda(df)
    with tab_cdss:
        render_cdss(df)
    with tab_model:
        render_model_insights()


main()
