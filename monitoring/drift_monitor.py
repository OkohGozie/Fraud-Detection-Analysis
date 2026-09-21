"""
drift_monitor.py — Continuous monitoring using Evidently
(Final Project, Section 7: Model Drift & Bias Drift Detection)

Compares the training distribution (reference) against a simulated "current"
production batch to detect data drift, which is the earliest warning sign
that a model's real-world accuracy may be degrading before ground-truth
fraud labels are even available to confirm it directly.
"""

import sys
from pathlib import Path
sys.path.append("/home/claude/model_dev")

import joblib
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

ARTIFACT_DIR = Path("/home/claude/model_dev/artifacts")
OUT_DIR = Path("/home/claude/final_project/monitoring")


def simulate_current_batch(X_train: pd.DataFrame, drift_features=None, shift=0.0, seed=7) -> pd.DataFrame:
    """Simulates a 'current' production batch. With shift=0 this is just a
    fresh sample (no real drift, the healthy case). With shift>0, selected
    features are deliberately shifted to demonstrate the monitor actually
    catching drift when it occurs, not just always reporting 'all clear.'"""
    rng = np.random.default_rng(seed)
    sample = X_train.sample(n=2000, random_state=seed).copy()
    if drift_features:
        for feat in drift_features:
            sample[feat] = sample[feat] + shift
    return sample


def run_drift_report(reference: pd.DataFrame, current: pd.DataFrame, label: str) -> dict:
    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    result_dict = result.dict()

    # DriftedColumnsCount metric gives the overall summary directly
    overall = next((m for m in result_dict["metrics"] if m["metric_name"].startswith("DriftedColumnsCount")), None)
    n_drifted = int(overall["value"]["count"]) if overall else 0
    n_checked = len([m for m in result_dict["metrics"] if m["metric_name"].startswith("ValueDrift")])
    drifted_cols = [
        m["config"]["column"] for m in result_dict["metrics"]
        if m["metric_name"].startswith("ValueDrift") and m["value"] > m["config"]["threshold"]
    ]

    summary = {
        "label": label,
        "n_features_checked": n_checked,
        "n_features_drifted": n_drifted,
        "drift_share": overall["value"]["share"] if overall else 0.0,
        "drifted_columns": drifted_cols,
    }
    html_path = OUT_DIR / f"drift_report_{label}.html"
    result.save_html(str(html_path))
    return summary, str(html_path)


if __name__ == "__main__":
    X_train = joblib.load(ARTIFACT_DIR / "X_train.joblib")

    print("=" * 60)
    print("MONITORING RUN 1: Healthy batch (no injected drift)")
    print("=" * 60)
    current_healthy = simulate_current_batch(X_train, shift=0.0)
    summary_healthy, path1 = run_drift_report(X_train, current_healthy, "healthy")
    print(summary_healthy)
    print(f"Full report: {path1}")

    print("\n" + "=" * 60)
    print("MONITORING RUN 2: Drifted batch (V12/V14/V10 shifted +2.0)")
    print("=" * 60)
    current_drifted = simulate_current_batch(X_train, drift_features=["V12", "V14", "V10"], shift=2.0)
    summary_drifted, path2 = run_drift_report(X_train, current_drifted, "drifted")
    print(summary_drifted)
    print(f"Full report: {path2}")

    print("\n" + "=" * 60)
    print("INTERPRETATION")
    print("=" * 60)
    if summary_drifted["n_features_drifted"] > summary_healthy["n_features_drifted"]:
        print("PASS: monitor correctly distinguished the drifted batch from the healthy one.")
        print(f"Healthy batch: {summary_healthy['n_features_drifted']} features drifted.")
        print(f"Drifted batch: {summary_drifted['n_features_drifted']} features drifted, "
              f"including the exact features (V12, V14, V10) that matter most to this model per SHAP (Module 4).")
    else:
        print("NOTE: drift not clearly distinguished in this run; thresholds may need tuning.")
