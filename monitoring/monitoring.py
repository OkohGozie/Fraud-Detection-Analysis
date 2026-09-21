"""
monitoring.py — Continuous drift and performance monitoring
(Final Project, Section 7: Model Drift & Bias Drift Detection).

Uses Evidently AI to compare a "reference" window (the training data the
model was validated against) to a "current" window (newer production
data). In this project, the training set stands in for the reference
window and the held-out test set stands in for a "current" window, since
we have no live production feed yet; this is the correct way to smoke-test
a monitoring pipeline before deployment, per Evidently's own recommended
practice (Evidently AI, 2026).

Run with: python monitoring.py
Produces: drift_report.html (full interactive report) and
          drift_summary.json (machine-readable summary for CI/CD gating)
"""

import joblib
import pandas as pd
import json
from pathlib import Path
from evidently import Report
from evidently.presets import DataDriftPreset

ARTIFACT_DIR = Path("/home/claude/model_dev/artifacts")
OUT_DIR = Path("/home/claude/final_project/monitoring")

def load_data():
    X_train = joblib.load(ARTIFACT_DIR / "X_train.joblib")
    X_test = joblib.load(ARTIFACT_DIR / "X_test.joblib")
    return X_train, X_test

def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame):
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    result.save_html(str(OUT_DIR / "drift_report.html"))
    result_dict = result.dict()
    return result_dict

def summarize(result_dict: dict) -> dict:
    """Extract a compact, CI/CD-friendly pass/fail summary."""
    metrics = result_dict.get("metrics", [])
    overall = next((m for m in metrics if m["metric_name"].startswith("DriftedColumnsCount")), None)
    dataset_drift = bool(overall["value"]["share"] > 0.5) if overall else False
    drifted_columns = [
        m["config"]["column"] for m in metrics
        if m["metric_name"].startswith("ValueDrift") and m["value"] > m["config"]["threshold"]
    ]
    summary = {
        "reference_rows": None,
        "current_rows": None,
        "dataset_drift_detected": dataset_drift,
        "drifted_columns": drifted_columns,
        "total_metrics_computed": len(metrics),
        "monitoring_tool": "Evidently AI",
        "monitored_windows": "train (reference) vs. held-out test (current) — pre-production smoke test",
        "action_if_drift_detected": "Halt automated retraining trigger; route to Data Science Lead for manual review before promoting any new model version (see Model Registry, Section 4).",
    }
    return summary

if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train, X_test = load_data()
    result_dict = run_drift_report(X_train, X_test)
    summary = summarize(result_dict)
    summary["reference_rows"] = len(X_train)
    summary["current_rows"] = len(X_test)
    with open(OUT_DIR / "drift_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nFull interactive report saved to: {OUT_DIR / 'drift_report.html'}")
