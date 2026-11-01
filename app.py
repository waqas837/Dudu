"""
Intelligent Diabetes Risk Predictor System
Primary routing & session hub (Streamlit).
"""

from __future__ import annotations

import streamlit as st

import db_helper

st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

db_helper.init_db()


def _init_session() -> None:
    defaults = {
        "authenticated": False,
        "user_id": None,
        "username": None,
        "role": None,
        "email": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def logout() -> None:
    for key in ("authenticated", "user_id", "username", "role", "email"):
        st.session_state[key] = False if key == "authenticated" else None


def render_sidebar() -> None:
    with st.sidebar:
        st.title("🩺 Diabetes Risk")
        if st.session_state.authenticated:
            st.markdown(f"**User:** {st.session_state.username}")
            st.markdown(f"**Role:** {st.session_state.role}")
            st.caption("Use the pages menu above to open your portal.")
            if st.button("Log Out", width="stretch", type="primary"):
                logout()
                st.rerun()
        else:
            st.info("Sign in to access Patient, Doctor, or Admin portals.")


def _complete_login(user: dict) -> None:
    st.session_state.authenticated = True
    st.session_state.user_id = user["user_id"]
    st.session_state.username = user["username"]
    st.session_state.role = user["role"]
    st.session_state.email = user["email"]
    st.rerun()


def login_tab() -> None:
    st.subheader("Login")
    username = st.text_input("Username", key="login_username")
    password = st.text_input("Password", type="password", key="login_password")
    if st.button("Sign In", width="stretch", type="primary", key="login_btn"):
        user = db_helper.authenticate_user(username, password)
        if user is None:
            st.error("Invalid username or password.")
        else:
            _complete_login(user)

    st.caption("Quick demo login")
    d1, d2, d3 = st.columns(3)
    if d1.button("Patient demo", key="demo_patient"):
        user = db_helper.authenticate_user("patient1", "patient123")
        if user:
            _complete_login(user)
    if d2.button("Doctor demo", key="demo_doctor"):
        user = db_helper.authenticate_user("doctor1", "doctor123")
        if user:
            _complete_login(user)
    if d3.button("Admin demo", key="demo_admin"):
        user = db_helper.authenticate_user("admin1", "admin123")
        if user:
            _complete_login(user)


def register_tab() -> None:
    st.subheader("Create Account")
    username = st.text_input("Username", key="reg_username")
    email = st.text_input("Email", key="reg_email")
    password = st.text_input("Password", type="password", key="reg_password")
    confirm = st.text_input("Confirm Password", type="password", key="reg_confirm")
    role = st.selectbox("Role", ["Patient", "Doctor", "Admin"], key="reg_role")
    if st.button("Register", width="stretch", key="reg_btn"):
        if not username or not email or not password:
            st.error("All fields are required.")
        elif password != confirm:
            st.error("Passwords do not match.")
        else:
            ok, message = db_helper.register_user(username, email, password, role)
            if ok:
                st.success(message)
            else:
                st.error(message)


def home_authenticated() -> None:
    import ml_engine

    role = st.session_state.role
    st.title("Intelligent Diabetes Risk Predictor")
    st.markdown(
        f"Signed in as **{st.session_state.username}** · role **{role}**."
    )
    status = ml_engine.model_status()
    if status["using_fallback"]:
        st.warning("ML model not loaded — predictions use the temporary fallback rule.")
    else:
        st.success(
            f"Production model ready: **{status['best_model_name']}** "
            f"({status['best_model_accuracy'] * 100:.2f}% accuracy)."
        )
    st.markdown("---")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### Patient Portal")
        st.write(
            "Predict risk, see factor explanations, track history, "
            "download CSV/PDF reports, and read education resources."
        )
    with col2:
        st.markdown("### Doctor Portal")
        st.write("EDA dashboards, KPIs, CDSS patient views, and model insights.")
    with col3:
        st.markdown("### Admin Portal")
        st.write("Dataset upload, official metrics, confusion matrix, and system logs.")

    st.info(
        "Open your portal from the **sidebar pages** list "
        "(Patient Portal / Doctor Portal / Admin Portal). "
        "Access is role-checked on each page."
    )

    if role == "Patient":
        st.success("Go to **Patient Portal** in the sidebar to run a risk assessment.")
    elif role == "Doctor":
        st.success("Go to **Doctor Portal** in the sidebar for clinical dashboards.")
    elif role == "Admin":
        st.success("Go to **Admin Portal** in the sidebar for dataset & metrics management.")

    with st.expander("Demo accounts (seeded on first run)"):
        st.code(
            "patient1 / patient123\n"
            "doctor1 / doctor123\n"
            "admin1 / admin123",
            language="text",
        )


def home_guest() -> None:
    st.title("Intelligent Diabetes Risk Predictor")
    st.markdown(
        "Role-based clinical decision-support platform for early diabetes risk "
        "prediction, patient tracking, and exploratory analysis."
    )
    tab_login, tab_register = st.tabs(["Login", "Register"])
    with tab_login:
        login_tab()
    with tab_register:
        register_tab()

    with st.expander("Demo accounts"):
        st.code(
            "patient1 / patient123\n"
            "doctor1 / doctor123\n"
            "admin1 / admin123",
            language="text",
        )


def main() -> None:
    _init_session()
    render_sidebar()
    if st.session_state.authenticated:
        home_authenticated()
    else:
        home_guest()


if __name__ == "__main__":
    main()
