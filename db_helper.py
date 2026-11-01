"""SQLite database interface and CRUD operations."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import bcrypt
import pandas as pd

DB_PATH = Path(__file__).resolve().parent / "diabetes_system.db"

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK(role IN ('Patient', 'Doctor', 'Admin')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS patient_records (
    record_id INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id INTEGER NOT NULL,
    pregnancies INTEGER DEFAULT 0,
    glucose REAL NOT NULL,
    blood_pressure REAL NOT NULL,
    skin_thickness REAL NOT NULL,
    insulin REAL NOT NULL,
    bmi REAL NOT NULL,
    diabetes_pedigree REAL NOT NULL,
    age INTEGER NOT NULL,
    prediction_result INTEGER CHECK(prediction_result IN (0, 1)),
    risk_probability REAL,
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS feedback_logs (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    user_role VARCHAR(20) NOT NULL,
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comments TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(50) NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create tables, seed demo accounts, and ensure official AI metrics exist."""
    with get_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        user_count = conn.execute("SELECT COUNT(*) AS c FROM users").fetchone()["c"]
        if user_count == 0:
            _seed_demo_users(conn)
        metric_count = conn.execute("SELECT COUNT(*) AS c FROM model_metrics").fetchone()["c"]
        if metric_count == 0:
            from ml_engine import OFFICIAL_METRICS

            for name, acc, prec, rec, f1 in OFFICIAL_METRICS:
                conn.execute(
                    """
                    INSERT INTO model_metrics
                        (model_name, accuracy, precision_score, recall_score, f1_score)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (name, acc, prec, rec, f1),
                )


def sync_official_metrics() -> None:
    """Replace model_metrics with the AI team's official evaluation numbers."""
    from ml_engine import OFFICIAL_METRICS

    replace_model_metrics(list(OFFICIAL_METRICS))


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        return False


def _seed_demo_users(conn: sqlite3.Connection) -> None:
    demos = [
        ("patient1", "patient1@demo.com", "patient123", "Patient"),
        ("doctor1", "doctor1@demo.com", "doctor123", "Doctor"),
        ("admin1", "admin1@demo.com", "admin123", "Admin"),
    ]
    for username, email, password, role in demos:
        conn.execute(
            """
            INSERT INTO users (username, email, password_hash, role)
            VALUES (?, ?, ?, ?)
            """,
            (username, email, _hash_password(password), role),
        )


def register_user(username: str, email: str, password: str, role: str) -> tuple[bool, str]:
    if role not in ("Patient", "Doctor", "Admin"):
        return False, "Invalid role selected."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    try:
        with get_connection() as conn:
            conn.execute(
                """
                INSERT INTO users (username, email, password_hash, role)
                VALUES (?, ?, ?, ?)
                """,
                (username.strip(), email.strip().lower(), _hash_password(password), role),
            )
        return True, "Account created successfully. Please log in."
    except sqlite3.IntegrityError as exc:
        msg = str(exc).lower()
        if "username" in msg:
            return False, "Username already exists."
        if "email" in msg:
            return False, "Email already registered."
        return False, "Username or email already exists."


def authenticate_user(username: str, password: str) -> Optional[dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (username.strip(),),
        ).fetchone()
    if row is None:
        return None
    if not verify_password(password, row["password_hash"]):
        return None
    return {
        "user_id": row["user_id"],
        "username": row["username"],
        "email": row["email"],
        "role": row["role"],
    }


def insert_patient_record(
    patient_id: int,
    pregnancies: int,
    glucose: float,
    blood_pressure: float,
    skin_thickness: float,
    insulin: float,
    bmi: float,
    diabetes_pedigree: float,
    age: int,
    prediction_result: int,
    risk_probability: float,
) -> int:
    with get_connection() as conn:
        cur = conn.execute(
            """
            INSERT INTO patient_records (
                patient_id, pregnancies, glucose, blood_pressure, skin_thickness,
                insulin, bmi, diabetes_pedigree, age, prediction_result, risk_probability
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                patient_id,
                pregnancies,
                glucose,
                blood_pressure,
                skin_thickness,
                insulin,
                bmi,
                diabetes_pedigree,
                age,
                prediction_result,
                risk_probability,
            ),
        )
        return int(cur.lastrowid)


def get_patient_records(patient_id: int) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT record_id, pregnancies, glucose, blood_pressure, skin_thickness,
                   insulin, bmi, diabetes_pedigree, age, prediction_result,
                   risk_probability, recorded_at
            FROM patient_records
            WHERE patient_id = ?
            ORDER BY recorded_at ASC
            """,
            conn,
            params=(patient_id,),
        )


def get_all_patient_records() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT pr.*, u.username AS patient_username, u.email AS patient_email
            FROM patient_records pr
            JOIN users u ON u.user_id = pr.patient_id
            ORDER BY pr.recorded_at DESC
            """,
            conn,
        )


def get_patient_list() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT DISTINCT u.user_id, u.username, u.email,
                   COUNT(pr.record_id) AS record_count
            FROM users u
            LEFT JOIN patient_records pr ON pr.patient_id = u.user_id
            WHERE u.role = 'Patient'
            GROUP BY u.user_id, u.username, u.email
            ORDER BY u.username
            """,
            conn,
        )


def insert_feedback(user_id: int, user_role: str, rating: int, comments: str) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO feedback_logs (user_id, user_role, rating, comments)
            VALUES (?, ?, ?, ?)
            """,
            (user_id, user_role, rating, comments.strip()),
        )


def get_all_feedback() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT f.feedback_id, u.username, f.user_role, f.rating,
                   f.comments, f.submitted_at
            FROM feedback_logs f
            JOIN users u ON u.user_id = f.user_id
            ORDER BY f.submitted_at DESC
            """,
            conn,
        )


def get_model_metrics() -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT metric_id, model_name, accuracy, precision_score,
                   recall_score, f1_score, trained_at
            FROM model_metrics
            ORDER BY trained_at DESC, accuracy DESC
            """,
            conn,
        )


def replace_model_metrics(rows: list[tuple[str, float, float, float, float]]) -> None:
    """Replace stored model metrics (used after admin re-evaluation)."""
    with get_connection() as conn:
        conn.execute("DELETE FROM model_metrics")
        for name, acc, prec, rec, f1 in rows:
            conn.execute(
                """
                INSERT INTO model_metrics
                    (model_name, accuracy, precision_score, recall_score, f1_score)
                VALUES (?, ?, ?, ?, ?)
                """,
                (name, acc, prec, rec, f1),
            )


def get_kpi_stats() -> dict[str, float]:
    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT
                COUNT(*) AS total_evaluated,
                COALESCE(AVG(glucose), 0) AS mean_glucose,
                COALESCE(AVG(bmi), 0) AS mean_bmi
            FROM patient_records
            """
        ).fetchone()
    return {
        "total_evaluated": int(row["total_evaluated"]),
        "mean_glucose": float(row["mean_glucose"]),
        "mean_bmi": float(row["mean_bmi"]),
    }


def get_recent_insert_logs(limit: int = 50) -> pd.DataFrame:
    with get_connection() as conn:
        return pd.read_sql_query(
            """
            SELECT pr.record_id, u.username AS patient_username, pr.glucose, pr.bmi,
                   pr.prediction_result, pr.risk_probability, pr.recorded_at
            FROM patient_records pr
            JOIN users u ON u.user_id = pr.patient_id
            ORDER BY pr.recorded_at DESC
            LIMIT ?
            """,
            conn,
            params=(limit,),
        )
