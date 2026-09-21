"""
fairness_analysis.py — Module 4 Section 6: Fairness Metrics & Mitigation

Honest limitation carried forward from Module 3's bias detection suite and
documented in the Module 3 RAID Log (Issue I1): this dataset has no
demographic fields (age, gender, region), since the original features are
PCA-anonymized at source. True Demographic Parity / Equalized Odds /
Disparate Impact Ratio analysis across protected groups is therefore not
possible on this dataset as-is.

What this script does instead, and states plainly as a proxy rather than a
substitute: computes the required fairness metrics (Demographic Parity
Difference, Equalized Odds Difference, Disparate Impact Ratio) using
transaction AMOUNT BAND as a stand-in grouping variable, since amount band
is the closest available proxy for differing customer segments (e.g., retail
vs. commercial account behavior) in this dataset. This demonstrates the
correct methodology and tooling (Fairlearn) so it can be directly re-applied
to real protected attributes once production KYC data is available under
NDPA-governed access, per the Module 2 Data Privacy Plan.
"""

import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from fairlearn.metrics import (MetricFrame, demographic_parity_difference,
                                 equalized_odds_difference, selection_rate,
                                 true_positive_rate, false_positive_rate)
from fairlearn.postprocessing import ThresholdOptimizer

ARTIFACT_DIR = Path("/home/claude/model_dev/artifacts")

model = joblib.load(ARTIFACT_DIR / "best_model.joblib")
X_test = joblib.load(ARTIFACT_DIR / "X_test.joblib")
y_test = joblib.load(ARTIFACT_DIR / "y_test.joblib")
X_train = joblib.load(ARTIFACT_DIR / "X_train.joblib")
y_train = pd.read_csv("/home/claude/model_dev/data_creditcard_clean.csv").loc[X_train.index, "Class"]

# Proxy grouping: amount band (documented limitation above)
def amount_band(amount):
    if amount < 10:
        return "small (<10)"
    elif amount < 200:
        return "mid (10-200)"
    else:
        return "large (200+)"

group_test = X_test["Amount"].apply(amount_band)
y_pred = model.predict(X_test)

print("=" * 65)
print("FAIRNESS ANALYSIS — Proxy grouping: transaction amount band")
print("KNOWN LIMITATION: no demographic fields available (Module 3, I1)")
print("=" * 65)

mf = MetricFrame(
    metrics={"selection_rate": selection_rate, "tpr": true_positive_rate, "fpr": false_positive_rate},
    y_true=y_test, y_pred=y_pred, sensitive_features=group_test,
)
print("\nPer-group metrics:")
print(mf.by_group)

dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=group_test)
try:
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=group_test)
except Exception as e:
    eod = None
    print(f"\n(Equalized odds difference could not be computed: {e} — likely due to a group having zero positive cases in this small test set)")

selection_rates = mf.by_group["selection_rate"]
dir_ratio = selection_rates.min() / selection_rates.max() if selection_rates.max() > 0 else float("nan")

print(f"\nDemographic Parity Difference: {dpd:.4f}")
if eod is not None:
    print(f"Equalized Odds Difference: {eod:.4f}")
print(f"Disparate Impact Ratio (min/max selection rate): {dir_ratio:.4f}")
print("(Common practitioner threshold: DIR < 0.80 signals potential adverse impact — the 'four-fifths rule')")

fairness_summary = {
    "demographic_parity_difference": float(dpd),
    "equalized_odds_difference": float(eod) if eod is not None else None,
    "disparate_impact_ratio": float(dir_ratio),
    "flag_raised": bool(dir_ratio < 0.80) if not np.isnan(dir_ratio) else None,
    "grouping_variable": "amount_band (PROXY — no demographic data available)",
}
pd.Series(fairness_summary).to_csv(ARTIFACT_DIR / "fairness_summary.csv")

# ---------- Bias Mitigation ----------
print("\n" + "=" * 65)
print("BIAS MITIGATION — ThresholdOptimizer (Fairlearn)")
print("=" * 65)
if fairness_summary["flag_raised"]:
    print("Disparate impact flag raised — applying ThresholdOptimizer to equalize selection rates across amount bands.")
    postprocess_est = ThresholdOptimizer(
        estimator=model, constraints="demographic_parity", predict_method="predict_proba", prefit=True,
    )
    group_train = X_train["Amount"].apply(amount_band)
    postprocess_est.fit(X_train, y_train, sensitive_features=group_train)
    y_pred_mitigated = postprocess_est.predict(X_test, sensitive_features=group_test)

    mf_mitigated = MetricFrame(
        metrics={"selection_rate": selection_rate}, y_true=y_test, y_pred=y_pred_mitigated, sensitive_features=group_test,
    )
    print("\nPer-group selection rate AFTER mitigation:")
    print(mf_mitigated.by_group)
    dpd_after = demographic_parity_difference(y_test, y_pred_mitigated, sensitive_features=group_test)
    print(f"\nDemographic Parity Difference after mitigation: {dpd_after:.4f} (was {dpd:.4f})")
else:
    print("No disparate impact flag raised at the 0.80 threshold on this proxy grouping.")
    print("Mitigation is documented as available (Fairlearn ThresholdOptimizer) but not")
    print("triggered here, since applying mitigation without a detected disparity would")
    print("trade away accuracy for no fairness benefit.")

print("\nFairness analysis complete. Summary saved to fairness_summary.csv")
