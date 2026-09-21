"""
generate_proxy_dataset.py

IMPORTANT / METHODOLOGY NOTE (see AI Disclosure and Section 1 of slide deck):
This sandboxed development environment cannot reach kaggle.com (network egress
is restricted to a fixed allowlist of domains for security reasons), so the
real ULB Credit Card Fraud Detection dataset could not be downloaded directly
during pipeline development. Per the assignment's own allowance
("you may supplement it with synthetic data containing noise to match the
characteristics of the original dataset"), this script generates a synthetic
proxy dataset that matches the REAL dataset's documented schema and class
distribution:
  - 284,807 rows (sampled down to 20,000 here for pipeline runtime speed;
    trivially changeable)
  - Columns: Time, V1-V28 (PCA-like, mean ~0, unit-ish variance), Amount, Class
  - Fraud rate: 0.172% (492 / 284,807 in the real dataset)

When running this pipeline for real submission/grading, replace the call to
generate_proxy_dataset() with pd.read_csv("creditcard.csv") after downloading
the real file from:
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
The rest of the pipeline (cleaning, transformation, validation, DAG, models)
is written to run unchanged against the real file, since it matches the same
schema exactly.
"""

import numpy as np
import pandas as pd

def generate_proxy_dataset(n_rows: int = 20000, fraud_rate: float = 0.00172, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_fraud = max(1, int(round(n_rows * fraud_rate)))
    n_legit = n_rows - n_fraud

    # Time: seconds elapsed across a 2-day window (172,800 seconds), matching real dataset span
    time_legit = rng.uniform(0, 172800, n_legit)
    time_fraud = rng.uniform(0, 172800, n_fraud)

    # V1-V28: PCA-like components. Legit transactions ~ N(0,1).
    # Fraud transactions get a shifted/scaled distribution on a subset of
    # components, mimicking the real dataset's known separability on components
    # like V14, V12, V10, V17 (documented in prior public analyses of this dataset).
    v_legit = rng.normal(loc=0.0, scale=1.0, size=(n_legit, 28))
    v_fraud = rng.normal(loc=0.0, scale=1.0, size=(n_fraud, 28))
    shifted_components = [9, 11, 13, 16]  # 0-indexed -> V10, V12, V14, V17
    for idx in shifted_components:
        v_fraud[:, idx] = rng.normal(loc=-4.5, scale=2.5, size=n_fraud)

    # Amount: legit transactions log-normal (typical retail spend); fraud
    # transactions skew toward smaller "testing" amounts with occasional spikes,
    # consistent with documented card-testing fraud behavior.
    amount_legit = np.round(rng.lognormal(mean=3.0, sigma=1.2, size=n_legit), 2)
    amount_fraud = np.round(
        np.where(rng.random(n_fraud) < 0.7,
                 rng.uniform(0.5, 50, n_fraud),
                 rng.uniform(200, 2000, n_fraud)), 2)

    legit_df = pd.DataFrame(v_legit, columns=[f"V{i+1}" for i in range(28)])
    legit_df.insert(0, "Time", time_legit)
    legit_df["Amount"] = amount_legit
    legit_df["Class"] = 0

    fraud_df = pd.DataFrame(v_fraud, columns=[f"V{i+1}" for i in range(28)])
    fraud_df.insert(0, "Time", time_fraud)
    fraud_df["Amount"] = amount_fraud
    fraud_df["Class"] = 1

    df = pd.concat([legit_df, fraud_df], ignore_index=True)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)  # shuffle

    # Inject realistic data quality issues for the pipeline to actually clean:
    # a small number of missing values and a small number of exact duplicates,
    # since real transaction feeds are never perfectly clean.
    missing_idx = rng.choice(df.index, size=int(0.001 * len(df)), replace=False)
    df.loc[missing_idx, "Amount"] = np.nan
    dup_rows = df.sample(n=int(0.002 * len(df)), random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df

if __name__ == "__main__":
    df = generate_proxy_dataset()
    df.to_csv("/home/claude/pipeline_project/data/raw/creditcard_proxy.csv", index=False)
    print(f"Generated {len(df)} rows, {df['Class'].sum()} fraud cases "
          f"({df['Class'].mean()*100:.3f}% fraud rate)")
    print(f"Missing values: {df.isna().sum().sum()}, Duplicates: {df.duplicated().sum()}")
