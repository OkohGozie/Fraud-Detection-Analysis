"""
bias_detection.py — Representation bias checks
(Module 3, Section 6: Implement a Bias Detection Suite).

Honest limitation, documented here and in the slide deck: the ULB Credit
Card Fraud dataset (and this proxy) contains no demographic fields
(no region, income, gender, age), because the original features are PCA-
anonymized specifically to prevent re-identification. This was already
flagged as Issue I1 in the Module 2 Technical RAID Log. True demographic
fairness auditing (Demographic Parity, Equalized Odds across protected
groups) is therefore NOT possible on this dataset as-is, and would only
become possible once a production deployment joins in KYC metadata
(Module 2 Data Dictionary) under proper NDPA-governed access.

What CAN be genuinely checked now, and what this script does, is
representation bias across the proxies available: transaction amount
bands and time-of-day bands. This checks whether the training data
adequately represents transaction patterns across the full operating
range, which matters because a model trained mostly on daytime, mid-size
transactions could underperform (and disproportionately misclassify) on
the off-hour or high-value transactions that matter most for the
prescriptive loss-exposure ranking (Module 2, Section 3).
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger("fraud_pipeline")


def representation_bias_report(df: pd.DataFrame) -> dict:
    report = {}

    # Amount-band representation: bin transactions into bands and check
    # whether any band is so sparse that a model would have little signal
    # to learn fraud patterns there.
    df = df.copy()
    df["amount_band"] = pd.cut(df["Amount"], bins=[0, 10, 50, 200, 1000, np.inf],
                                 labels=["<10", "10-50", "50-200", "200-1000", "1000+"])
    amount_dist = df.groupby("amount_band", observed=True).agg(
        n_transactions=("Amount", "count"),
        fraud_rate=("Class", "mean"),
    )
    report["amount_band_distribution"] = amount_dist.to_dict("index")

    # Time-of-day representation: check whether off-hour transactions
    # (a known higher-risk window in fraud literature) are represented
    # in sufficient volume to train on, not just theoretically present.
    hour_dist = df.groupby("hour_of_day", observed=True).agg(
        n_transactions=("Amount", "count"),
        fraud_rate=("Class", "mean"),
    )
    report["hour_of_day_distribution"] = hour_dist.to_dict("index")

    # Flag any band below a minimum representation threshold (fewer than
    # 30 examples is too sparse for reliable model learning in that band).
    sparse_amount_bands = amount_dist[amount_dist["n_transactions"] < 30].index.tolist()
    sparse_hours = hour_dist[hour_dist["n_transactions"] < 30].index.tolist()

    report["sparse_amount_bands"] = [str(b) for b in sparse_amount_bands]
    report["sparse_hours"] = [int(h) for h in sparse_hours]
    report["bias_flag_raised"] = bool(sparse_amount_bands or sparse_hours)
    report["known_limitation"] = (
        "No demographic fields are available in this dataset (PCA-anonymized "
        "by source). True Demographic Parity / Equalized Odds auditing across "
        "protected groups requires production KYC data under NDPA-governed "
        "access (see Module 2 RAID Log, Issue I1)."
    )

    return report


if __name__ == "__main__":
    df = pd.read_csv("/home/claude/pipeline_project/data/processed/creditcard_clean.csv")
    result = representation_bias_report(df)

    print("=" * 60)
    print("BIAS DETECTION SUITE — REPRESENTATION BIAS REPORT")
    print("=" * 60)
    print("\nAmount band distribution (n transactions, fraud rate):")
    for band, stats in result["amount_band_distribution"].items():
        print(f"  {band:>10}: n={stats['n_transactions']:>6}  fraud_rate={stats['fraud_rate']:.4f}")

    print(f"\nSparse amount bands (n<30): {result['sparse_amount_bands'] or 'None'}")
    print(f"Sparse hour bins (n<30): {result['sparse_hours'] or 'None'}")
    print(f"\nBias flag raised: {result['bias_flag_raised']}")
    print(f"\nKnown limitation:\n  {result['known_limitation']}")
