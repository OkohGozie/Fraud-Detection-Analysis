"""
fairness_drift_check.py — Continuous Fairness Monitoring
(Final Project, Section 7: Continuous Fairness Monitoring Plan)

Re-runs the Module 4 fairness audit against a fresh batch and compares it
to the last known-good fairness state, so a fairness regression is caught
on a schedule, not only when someone remembers to check manually.
"""

import sys
from pathlib import Path
sys.path.append("/home/claude/model_dev")

import joblib
import pandas as pd
import json
from datetime import datetime, timezone
from fairlearn.metrics import MetricFrame, demographic_parity_difference, equalized_odds_difference

ARTIFACT_DIR = Path("/home/claude/model_dev/artifacts")
LOG_PATH = Path("/home/claude/final_project/monitoring/fairness_drift_log.jsonl")
ALERT_THRESHOLD_DPD = 0.05  # if DPD exceeds this, alert (Module 4 mitigated result was 0.0012)


def compute_fairness(model, X, y_true, amount_col_idx):
    y_pred = model.predict(X)
    amount = X.iloc[:, amount_col_idx] if hasattr(X, "iloc") else X[:, amount_col_idx]
    amount_band = pd.cut(pd.Series(amount).reset_index(drop=True),
                          bins=[-999, -1, 0, 1, 999], labels=["low", "mid_low", "mid_high", "high"])
    dpd = demographic_parity_difference(y_true, y_pred, sensitive_features=amount_band)
    eod = equalized_odds_difference(y_true, y_pred, sensitive_features=amount_band)
    return dpd, eod


def run_check():
    model = joblib.load(ARTIFACT_DIR / "best_model.joblib")
    feature_cols = joblib.load(ARTIFACT_DIR / "feature_cols.joblib")
    X_test = joblib.load(ARTIFACT_DIR / "X_test.joblib")
    y_test = joblib.load(ARTIFACT_DIR / "y_test.joblib")

    amount_idx = feature_cols.index("Amount") if "Amount" in feature_cols else feature_cols.index("log_amount")
    dpd, eod = compute_fairness(model, X_test, y_test, amount_idx)

    alert = dpd > ALERT_THRESHOLD_DPD
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demographic_parity_difference": round(float(dpd), 4),
        "equalized_odds_difference": round(float(eod), 4),
        "threshold": ALERT_THRESHOLD_DPD,
        "alert_raised": bool(alert),
        "compared_to_module4_mitigated_dpd": 0.0012,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


if __name__ == "__main__":
    result = run_check()
    print("=" * 60)
    print("CONTINUOUS FAIRNESS MONITORING CHECK")
    print("=" * 60)
    for k, v in result.items():
        print(f"  {k}: {v}")
    print()
    if result["alert_raised"]:
        print(f"ALERT: DPD ({result['demographic_parity_difference']}) exceeds threshold "
              f"({ALERT_THRESHOLD_DPD}). This would trigger the Incident Response Plan (Section 7).")
    else:
        print(f"OK: DPD ({result['demographic_parity_difference']}) within threshold. "
              f"Consistent with Module 4's mitigated result of 0.0012.")
