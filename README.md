# Dudu — Intelligent Diabetes Risk Predictor

Full-stack Streamlit web app for diabetes risk prediction with role-based access (Patient, Doctor, Admin), SQLite persistence, and a production Logistic Regression model.

## Features

- Secure register/login with bcrypt password hashing
- **Patient portal:** clinical inputs, live prediction, risk-factor explanation, history charts, CSV/PDF reports, education, feedback
- **Doctor portal:** EDA dashboards, KPIs, clinical decision support, model insights
- **Admin portal:** CSV dataset upload + cleaning, official model metrics, confusion matrix, system logs
- ML artifacts: `modals/best_diabetes_model.pkl` + `modals/scaler.pkl` (Logistic Regression, ~74.46% accuracy)

## Stack

- Python / Streamlit
- SQLite (`diabetes_system.db` created at runtime)
- scikit-learn, pandas, matplotlib, seaborn, plotly, fpdf2, bcrypt

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py
```

Open http://localhost:8501

### Demo accounts

| Role    | Username  | Password    |
|---------|-----------|-------------|
| Patient | patient1  | patient123  |
| Doctor  | doctor1   | doctor123   |
| Admin   | admin1    | admin123    |

## Project structure

```text
app.py                     # Auth + session hub
db_helper.py               # SQLite CRUD
utils.py                   # Cleaning + education content
ml_engine.py               # Model load / predict / explain
report_helper.py           # CSV/PDF reports
pages/
  1_Patient_Portal.py
  2_Doctor_Portal.py
  3_Admin_Portal.py
modals/
  best_diabetes_model.pkl
  scaler.pkl
  finalnotebook.ipynb
```

## Tests

```bash
python smoke_test.py
python ui_apptest.py
```

## Disclaimer

This system is a clinical decision-support aid only. Clinical judgment takes precedence.
