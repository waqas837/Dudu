"""CSV/PDF report generation helpers."""

from __future__ import annotations

from datetime import datetime
from io import BytesIO

import pandas as pd
from fpdf import FPDF


def _pdf_safe(text: str) -> str:
    """Core Helvetica is latin-1; strip/replace unsupported glyphs."""
    replacements = {
        "≥": ">=",
        "≤": "<=",
        "–": "-",
        "—": "-",
        "·": "-",
        "µ": "u",
        "“": '"',
        "”": '"',
        "’": "'",
    }
    out = str(text)
    for src, dst in replacements.items():
        out = out.replace(src, dst)
    return out.encode("latin-1", errors="replace").decode("latin-1")


class _ReportPDF(FPDF):
    def write_line(self, text: str, size: int = 10, bold: bool = False) -> None:
        self.set_x(self.l_margin)
        style = "B" if bold else ""
        self.set_font("Helvetica", style, size)
        self.multi_cell(self.epw, 5, _pdf_safe(text))


def build_patient_pdf(
    username: str,
    history_df: pd.DataFrame,
    latest_prediction: dict | None = None,
    advice: list[str] | None = None,
    explanations: list[dict] | None = None,
) -> bytes:
    """Create a simple patient health report PDF."""
    pdf = _ReportPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    pdf.write_line("Diabetes Risk Report", size=16, bold=True)
    pdf.write_line(f"Patient: {username}", size=11)
    pdf.write_line(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", size=11)
    pdf.ln(2)

    if latest_prediction:
        pdf.write_line("Latest Assessment", size=13, bold=True)
        label = "High Risk" if latest_prediction.get("prediction") == 1 else "Low Risk"
        pdf.write_line(
            f"Result: {label}  |  Confidence: {latest_prediction.get('confidence', 0):.1f}%",
            size=11,
        )
        pdf.ln(1)

    if advice:
        pdf.write_line("Personalized Recommendations", size=13, bold=True)
        for tip in advice:
            pdf.write_line(f"- {tip}", size=10)
        pdf.ln(1)

    if explanations:
        pdf.write_line("Key Risk Factors (this assessment)", size=13, bold=True)
        for row in explanations[:5]:
            pdf.write_line(
                f"{row['feature']}: value={row['value']:.2f} ({row['direction']})",
                size=10,
            )
        pdf.ln(1)

    pdf.write_line("Historical Records", size=13, bold=True)
    if history_df is None or history_df.empty:
        pdf.write_line("No historical records available.", size=10)
    else:
        cols = [
            c
            for c in [
                "recorded_at",
                "glucose",
                "bmi",
                "age",
                "prediction_result",
                "risk_probability",
            ]
            if c in history_df.columns
        ]
        for _, row in history_df[cols].tail(12).iterrows():
            pred = row.get("prediction_result", "")
            label = {0: "Low", 1: "High"}.get(pred, str(pred))
            line = (
                f"{row.get('recorded_at', '')} | Glu={row.get('glucose', '')} | "
                f"BMI={row.get('bmi', '')} | Age={row.get('age', '')} | "
                f"Risk={label} | Conf={row.get('risk_probability', '')}"
            )
            pdf.write_line(line, size=9)

    pdf.ln(3)
    pdf.write_line(
        "Disclaimer: This report is a clinical decision-support aid only. "
        "It does not replace professional medical judgment or diagnosis.",
        size=9,
    )

    out = pdf.output()
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return BytesIO(out).getvalue() if hasattr(out, "encode") else bytes(out)


def build_metrics_csv(metrics_df: pd.DataFrame) -> bytes:
    return metrics_df.to_csv(index=False).encode("utf-8")
