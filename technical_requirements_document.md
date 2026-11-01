# 📋 Technical Requirements & System Architecture Document

**Project Title:** Intelligent Diabetes Risk Predictor System  
**Target Environment:** Web Application (Streamlit / Python Stack)  
**Database Engine:** SQLite 3  
**Architecture Model:** Modular Multi-Tier (Frontend UI, Backend Services, SQLite DB, ML Engine Hook)  

---

## 1. Executive Summary & Architecture Overview

The **Intelligent Diabetes Risk Predictor System** is a machine learning-enabled clinical decision support platform designed to provide early risk prediction, patient tracking, and exploratory data analysis. 

The developer's goal is to construct:
1. **Frontend Layer:** Responsive, role-based User Interfaces for **Patients**, **Doctors/Clinicians**, and **Administrators**.
2. **Backend Service Layer:** Data processing pipelines, authentication routines, session handlers, and SQLite query managers.
3. **Database Layer (SQLite):** Persistent storage for user credentials, longitudinal health records, evaluation metrics, and feedback.
4. **AI/ML Connection Hooks:** Standardized interface functions where the Machine Learning model (`pickle`/`joblib` pipelines) can be seamlessly integrated.

---

## 2. Database Design & Schema Specifications (SQLite)

The SQLite database (`diabetes_system.db`) must be initialized with foreign key support enabled (`PRAGMA foreign_keys = ON;`).

```sql
-- 1. USERS TABLE: Manages authentication & role-based access
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK(role IN ('Patient', 'Doctor', 'Admin')) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. PATIENT_RECORDS TABLE: Stores medical attributes & longitudinal tracking data
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
    prediction_result INTEGER CHECK(prediction_result IN (0, 1)), -- 0: Low Risk, 1: High Risk
    risk_probability REAL, -- Model confidence score (0.0 to 100.0%)
    recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (patient_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 3. FEEDBACK_LOGS TABLE: Stores user feedback and system performance reviews
CREATE TABLE IF NOT EXISTS feedback_logs (
    feedback_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    user_role VARCHAR(20) NOT NULL,
    rating INTEGER CHECK(rating BETWEEN 1 AND 5),
    comments TEXT,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
);

-- 4. MODEL_METRICS TABLE: Tracks admin model evaluations & retraining benchmarks
CREATE TABLE IF NOT EXISTS model_metrics (
    metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_name VARCHAR(50) NOT NULL,
    accuracy REAL NOT NULL,
    precision_score REAL NOT NULL,
    recall_score REAL NOT NULL,
    f1_score REAL NOT NULL,
    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

---

## 3. Modular Folder & File Structure

The developer must enforce a clean separation of concerns:

```text
Diabetes_Project/
│
├── database.db                # SQLite database file
├── app.py                     # Primary routing & session hub
├── db_helper.py               # SQLite database interface & CRUD operations
├── utils.py                   # Data cleaning, normalization & utility functions
├── ml_engine.py               # ML Model loading & inference wrapper (AI Hook)
│
└── pages/                     # Streamlit Multi-Page Modules
    ├── 1_Patient_Portal.py    # Patient form, dynamic advice, longitudinal charts
    ├── 2_Doctor_Portal.py     # EDA dashboards & clinical decision support
    └── 3_Admin_Portal.py      # Retraining logs, dataset upload & model metrics
```

---

## 4. Component-Wise Functional Requirements

### 4.1 Authentication & User Session Management (`app.py`)
- **Login / Register System:** Role-based login supporting Password Hashing (`bcrypt` or `hashlib`).
- **Session Routing:** Store `st.session_state['user_id']`, `st.session_state['username']`, and `st.session_state['role']`.
- Dynamic sidebar navigation showing current session profile and a **Log Out** trigger.

---

### 4.2 Patient Portal (`pages/1_Patient_Portal.py`)
- **Clinical Feature Inputs:**
  - `Pregnancies`: Integer counter ($0 \rightarrow 20$)
  - `Glucose`: Numeric slider ($0 \rightarrow 200 \text{ mg/dL}$)
  - `Blood Pressure`: Numeric slider ($0 \rightarrow 150 \text{ mm Hg}$)
  - `Skin Thickness`: Numeric slider ($0 \rightarrow 100 \text{ mm}$)
  - `Insulin`: Numeric slider ($0 \rightarrow 900 \text{ }\mu\text{U/ml}$)
  - `BMI`: Double precision slider ($0.0 \rightarrow 70.0$)
  - `Diabetes Pedigree Function`: Double precision input ($0.00 \rightarrow 3.00$)
  - `Age`: Integer slider ($1 \rightarrow 100 \text{ years}$)
- **Inference Trigger Button:** Runs `ml_engine.predict_risk()` and logs inputs & results directly into the `patient_records` SQLite table.
- **Risk Assessment Panel:** Displays binary outcome (High/Low Risk) with confidence percentage.
- **Personalized Advice Module:** Dynamic medical recommendations based on Glucose & BMI threshold rules.
- **Longitudinal Charting:** Interactive trend line chart (`Glucose` & `BMI` vs `Time`) using historical records fetched from SQLite.
- **Report Exporter:** Button generating downloadable CSV report of the patient's personal history.
- **Feedback Form:** Form submitting star rating (1–5) and review text to `feedback_logs`.

---

### 4.3 Doctor / Healthcare Provider Portal (`pages/2_Doctor_Portal.py`)
- **Clinical EDA Dashboard:**
  - Dynamic KPI metric cards: *Total Evaluated Patients*, *Mean Glucose*, *Mean BMI*.
  - Histograms: Glucose level distribution across positive/negative outcomes.
  - Boxplots: Outlier detection in BMI and Insulin fields.
  - Heatmap: Inter-feature correlation matrix using standard plotting libraries (`matplotlib`/`seaborn`).
- **Clinical Decision Support System (CDSS):**
  - Searchable dropdown selecting patient profiles.
  - Tabular view of patient parameters alongside model risk predictions.
  - Mandatory disclaimer notice: *"System designed as a clinical decision-support aid. Clinical judgment takes precedence."*

---

### 4.4 Administrator Portal (`pages/3_Admin_Portal.py`)
- **Dataset Management:** File uploader interface (`.csv` format) for updating training data. Automatically run missing-value replacement routines (imputing 0s with column medians).
- **Model Benchmark Matrix:** Display tabular evaluation comparison across models:
  - Logistic Regression
  - Decision Tree Classifier
  - Support Vector Machine (SVM)
  - Multi-Layer Perceptron (Neural Network)
- **Log Monitor:** Table view listing system feedback entries and data insertion logs.

---

## 5. Backend Logic & AI Connection Contract

To ensure the AI developer can easily connect their trained model, the backend developer must create a clean interface inside `ml_engine.py`:

```python
# ml_engine.py
import joblib
import numpy as np

MODEL_PATH = 'diabetes_model.pkl'
SCALER_PATH = 'scaler.pkl'

def load_artifacts():
    """Loads the pre-trained ML model and standard scaler."""
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except Exception as e:
        return None, None

def predict_risk(features_list):
    """
    Interface function for AI predictions.
    
    Parameters:
        features_list (list): [pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]
        
    Returns:
        tuple: (prediction_class [0 or 1], confidence_percentage [float])
    """
    model, scaler = load_artifacts()
    
    if model is not None and scaler is not None:
        raw_data = np.array([features_list])
        scaled_data = scaler.transform(raw_data)
        prediction = int(model.predict(scaled_data)[0])
        probabilities = model.predict_proba(scaled_data)[0]
        confidence = float(np.max(probabilities) * 100)
        return prediction, confidence
    else:
        # Fallback simulation if model files are missing during UI testing
        glucose = features_list[1]
        simulated_pred = 1 if glucose > 125 else 0
        return simulated_pred, 85.0
```

---

## 6. Acceptance Criteria for Developer Handover

1. **Database Persistence:** All data submitted through the UI (User signups, Health records, Feedback) must correctly read/write to `diabetes_system.db`.
2. **Page Navigation:** Multi-page layout operating seamlessly via Streamlit sidebar options.
3. **Data Preprocessing Placeholder:** Automated cleaning functions in `utils.py` that replace invalid zero values (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) with median values.
4. **AI Ready:** `ml_engine.py` successfully returns model predictions when `.pkl` files are placed in the root directory.