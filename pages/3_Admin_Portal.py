"""Admin: dataset upload, model metrics, confusion matrix, logs."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

import db_helper
import ml_engine
import report_helper
from utils import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    clean_uploaded_dataset,
    load_training_dataset,
    save_training_dataset,
)

st.set_page_config(page_title="Admin Portal", page_icon="🛠️", layout="wide")
db_helper.init_db()


def require_admin() -> None:
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
        if st.button("Admin demo login", key="portal_demo_admin"):
            user = db_helper.authenticate_user("admin1", "admin123")
            if user:
                st.session_state.authenticated = True
                st.session_state.user_id = user["user_id"]
                st.session_state.username = user["username"]
                st.session_state.role = user["role"]
                st.session_state.email = user["email"]
                st.rerun()
        st.stop()
    if st.session_state.get("role") != "Admin":
        st.error("This portal is only available to Admin accounts.")
        st.stop()


def render_ml_hook_status() -> None:
    status = ml_engine.model_status()
    if status["using_fallback"]:
        st.info(
            "AI artifacts not loaded. Expected files: "
            "`modals/best_diabetes_model.pkl` and `modals/scaler.pkl`."
        )
    else:
        st.success(
            f"Loaded **{status['best_model_name']}** from `{status['model_path']}` "
            f"(accuracy {status['best_model_accuracy'] * 100:.2f}%). "
            "Logistic Regression was selected as the best model by the AI team."
        )


def evaluate_models(df: pd.DataFrame) -> list[tuple[str, float, float, float, float]]:
    """Optional re-benchmark after dataset upload (admin retrain support)."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError("Training CSV must include an Outcome column for benchmarking.")

    work = df[FEATURE_COLUMNS + [TARGET_COLUMN]].dropna().copy()
    X = work[FEATURE_COLUMNS]
    y = work[TARGET_COLUMN].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Support Vector Machine (SVM)": SVC(probability=True, random_state=42),
        "Neural Networks": MLPClassifier(
            hidden_layer_sizes=(32, 16), max_iter=500, random_state=42
        ),
    }

    rows = []
    for name, model in models.items():
        # Match AI notebook convention: LR/DT on raw, SVM/NN on scaled
        if name in {"Support Vector Machine (SVM)", "Neural Networks"}:
            model.fit(X_train_s, y_train)
            preds = model.predict(X_test_s)
        else:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
        rows.append(
            (
                name,
                float(accuracy_score(y_test, preds)),
                float(precision_score(y_test, preds, zero_division=0)),
                float(recall_score(y_test, preds, zero_division=0)),
                float(f1_score(y_test, preds, zero_division=0)),
            )
        )
    return rows


def render_dataset_management() -> None:
    st.subheader("Dataset Management")
    st.caption(
        "Upload a CSV (Kaggle diabeticprediction / Pima-style). "
        "Invalid medical zeros are replaced with column medians."
    )
    uploaded = st.file_uploader("Upload training CSV", type=["csv"])
    existing = load_training_dataset()
    if existing is not None:
        st.caption(f"Current stored training rows: {len(existing)}")
        st.dataframe(existing.head(10), width="stretch", hide_index=True)

    if uploaded is None:
        return

    raw = pd.read_csv(uploaded)
    cleaned, message = clean_uploaded_dataset(raw)
    st.write(message)
    st.dataframe(cleaned.head(15), width="stretch", hide_index=True)

    if st.button("Save cleaned dataset & re-benchmark", type="primary"):
        path = save_training_dataset(cleaned)
        st.success(f"Saved to {path}")

        if TARGET_COLUMN in cleaned.columns:
            try:
                rows = evaluate_models(cleaned)
                db_helper.replace_model_metrics(rows)
                st.success("Model benchmark matrix refreshed and stored in SQLite.")
            except Exception as exc:
                st.warning(f"Dataset saved, but benchmarking failed: {exc}")
        else:
            st.info("No Outcome column — metrics not recalculated.")


def render_metrics() -> None:
    st.subheader("Model Benchmark Matrix")
    st.markdown(
        "Logistic Regression achieved the highest accuracy (~**74.46%**), "
        "so it is the production model used by the web app."
    )

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Restore official AI metrics", key="restore_official"):
            db_helper.sync_official_metrics()
            st.success("Official AI metrics restored.")
            st.rerun()

    metrics = db_helper.get_model_metrics()
    if metrics.empty:
        st.info("No metrics stored yet.")
        return

    view = metrics.copy()
    for col in ("accuracy", "precision_score", "recall_score", "f1_score"):
        view[col] = (view[col] * 100).round(2)
    view = view.rename(
        columns={
            "model_name": "Model",
            "accuracy": "Accuracy %",
            "precision_score": "Precision %",
            "recall_score": "Recall %",
            "f1_score": "F1 %",
            "trained_at": "Evaluated At",
        }
    )
    display = view[["Model", "Accuracy %", "Precision %", "Recall %", "F1 %", "Evaluated At"]]
    st.dataframe(display, width="stretch", hide_index=True)

    st.download_button(
        "Export metrics CSV",
        data=report_helper.build_metrics_csv(display),
        file_name="model_metrics.csv",
        mime="text/csv",
        key="metrics_csv",
    )

    st.markdown("### Confusion Matrix (best model on stored training data)")
    train_df = load_training_dataset()
    if train_df is None or "Outcome" not in train_df.columns:
        st.info(
            "Upload and save a labeled training CSV to compute a live confusion matrix "
            "for the production model."
        )
        return

    result = ml_engine.evaluate_confusion_on_dataset(train_df)
    if result is None:
        st.warning("Could not compute confusion matrix.")
        return

    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    sns.heatmap(
        result["matrix"],
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Pred Low", "Pred High"],
        yticklabels=["Actual Low", "Actual High"],
        ax=ax,
    )
    ax.set_title("Confusion Matrix — Logistic Regression")
    st.pyplot(fig)
    plt.close(fig)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("True Negatives", result["tn"])
    m2.metric("False Positives", result["fp"])
    m3.metric("False Negatives", result["fn"])
    m4.metric("True Positives", result["tp"])


def render_logs() -> None:
    st.subheader("System Logs")
    tab_fb, tab_ins = st.tabs(["Feedback entries", "Data insertion logs"])
    with tab_fb:
        feedback = db_helper.get_all_feedback()
        if feedback.empty:
            st.info("No feedback yet.")
        else:
            st.dataframe(feedback, width="stretch", hide_index=True)
    with tab_ins:
        inserts = db_helper.get_recent_insert_logs()
        if inserts.empty:
            st.info("No patient records inserted yet.")
        else:
            show = inserts.copy()
            show["risk_label"] = show["prediction_result"].map({0: "Low Risk", 1: "High Risk"})
            st.dataframe(show, width="stretch", hide_index=True)


def main() -> None:
    require_admin()
    st.title("Administrator Portal")
    st.caption(f"Logged in as {st.session_state.username}")
    render_ml_hook_status()

    tab_data, tab_metrics, tab_logs = st.tabs(
        ["Dataset Management", "Model Metrics", "Log Monitor"]
    )
    with tab_data:
        render_dataset_management()
    with tab_metrics:
        render_metrics()
    with tab_logs:
        render_logs()


main()
