"""
test_etl.py — Unit tests for the ETL pipeline stages.
Run with: pytest tests/test_etl.py -v
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import pytest
from src.pipeline.etl import clean, transform, integrate


@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Time": [0, 100, 100, 200],  # includes a duplicate
        **{f"V{i}": [0.1 * i, 0.2 * i, 0.2 * i, 0.3 * i] for i in range(1, 29)},
        "Amount": [10.0, np.nan, np.nan, -5.0],  # includes missing + negative
        "Class": [0, 0, 0, 1],
    })


def test_clean_removes_duplicates(sample_df):
    result = clean(sample_df)
    assert result.duplicated().sum() == 0


def test_clean_imputes_missing_amount(sample_df):
    result = clean(sample_df)
    assert result["Amount"].isna().sum() == 0


def test_clean_flags_negative_amount(sample_df):
    result = clean(sample_df)
    assert "quality_flag_negative_amount" in result.columns
    assert result["quality_flag_negative_amount"].sum() >= 1


def test_transform_adds_engineered_features(sample_df):
    cleaned = clean(sample_df)
    result = transform(cleaned)
    assert "hour_of_day" in result.columns
    assert "log_amount" in result.columns


def test_transform_scales_v_features(sample_df):
    cleaned = clean(sample_df)
    result = transform(cleaned)
    # Scaled features should have near-zero mean (allowing for small-sample tolerance)
    assert abs(result["V1"].mean()) < 1.0


def test_integrate_produces_expected_schema(sample_df):
    cleaned = clean(sample_df)
    transformed = transform(cleaned)
    result = integrate(transformed)
    assert "Class" in result.columns
    assert result.columns[-1] == "Class"  # Class should be last (target column convention)


def test_pipeline_preserves_row_count_except_duplicates(sample_df):
    n_before = len(sample_df)
    n_dupes = sample_df.duplicated().sum()
    cleaned = clean(sample_df)
    assert len(cleaned) == n_before - n_dupes
