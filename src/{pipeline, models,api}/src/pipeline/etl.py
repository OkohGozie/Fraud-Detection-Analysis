"""
etl.py — Core ETL functions for the fraud detection pipeline.
Each function is a discrete, testable pipeline stage, called by both the
Prefect flow (dags/fraud_pipeline_flow.py) and the unit tests (tests/).
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logger = logging.getLogger("fraud_pipeline")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

RAW_PATH = Path("/home/claude/pipeline_project/data/raw/creditcard_proxy.csv")
PROCESSED_PATH = Path("/home/claude/pipeline_project/data/processed/creditcard_clean.csv")


def ingest(path: Path = RAW_PATH) -> pd.DataFrame:
    """Stage 1: Ingestion. Reads raw transaction data from source."""
    df = pd.read_csv(path)
    logger.info(f"INGEST: loaded {len(df)} rows from {path}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 2: Cleaning. Handles missing values, duplicates, and data quality issues."""
    n_before = len(df)

    # Duplicates: exact duplicate transactions are dropped (a transaction feed
    # should not contain the identical row twice; if it does, it is almost
    # certainly a feed error, not two genuine transactions).
    n_dupes = df.duplicated().sum()
    df = df.drop_duplicates().reset_index(drop=True)

    # Missing values: Amount is the only field with real-world missingness
    # risk (a feed error), imputed with the median rather than mean because
    # Amount is right-skewed (a few very large transactions would distort a
    # mean-based imputation).
    n_missing = df["Amount"].isna().sum()
    median_amount = df["Amount"].median()
    df["Amount"] = df["Amount"].fillna(median_amount)

    # Data quality: negative amounts are not physically valid for this
    # dataset (a purchase transaction cannot have negative value here) and are
    # flagged, not silently dropped, since a negative amount in a real feed
    # is itself a signal worth investigating, not just noise to remove.
    n_negative = (df["Amount"] < 0).sum()
    df["quality_flag_negative_amount"] = df["Amount"] < 0

    logger.info(f"CLEAN: {n_dupes} duplicates removed, {n_missing} missing Amount values "
                f"imputed with median ({median_amount:.2f}), {n_negative} negative amounts flagged. "
                f"Rows: {n_before} -> {len(df)}")
    return df


def transform(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 3: Transformation. Feature engineering for the modeling layer."""
    # Time-of-day feature, since a raw "seconds elapsed" field is not directly
    # meaningful to a model; time-of-day is a documented behavioral signal in
    # fraud detection (unusual-hour transactions carry more risk).
    df["hour_of_day"] = (df["Time"] % 86400) // 3600

    # Log-transform Amount: fraud detection literature consistently applies
    # a log transform to transaction amount because raw amount is heavily
    # right-skewed, which can dominate distance- and gradient-based models
    # if left untransformed.
    df["log_amount"] = np.log1p(df["Amount"])

    # Scale V-features to zero mean / unit variance for modeling stability,
    # explicitly excluding Class (the target) and the flags/derived columns.
    v_cols = [c for c in df.columns if c.startswith("V")]
    df[v_cols] = (df[v_cols] - df[v_cols].mean()) / df[v_cols].std()

    logger.info(f"TRANSFORM: engineered hour_of_day, log_amount; scaled {len(v_cols)} V-features")
    return df


def integrate(df: pd.DataFrame) -> pd.DataFrame:
    """Stage 4: Integration. In production, this stage would join the
    transaction table above against KYC/account metadata and device metadata
    (per the Module 2 Data Dictionary). Since this pipeline runs against the
    public proxy dataset only, integration here is a schema-alignment and
    column-ordering step so the output matches the exact schema a joined,
    production dataset would use downstream."""
    ordered_cols = (["Time", "hour_of_day"]
                     + [c for c in df.columns if c.startswith("V")]
                     + ["Amount", "log_amount", "quality_flag_negative_amount", "Class"])
    df = df[ordered_cols]
    logger.info(f"INTEGRATE: schema aligned to {len(ordered_cols)} columns for downstream modeling")
    return df


def save(df: pd.DataFrame, path: Path = PROCESSED_PATH) -> None:
    """Stage 5 (pre-validation): persist the processed dataset."""
    df.to_csv(path, index=False)
    logger.info(f"SAVE: wrote {len(df)} rows to {path}")


def run_pipeline() -> pd.DataFrame:
    """Runs the full ingest -> clean -> transform -> integrate -> save sequence."""
    df = ingest()
    df = clean(df)
    df = transform(df)
    df = integrate(df)
    save(df)
    return df


if __name__ == "__main__":
    run_pipeline()
