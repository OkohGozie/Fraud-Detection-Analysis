"""
fraud_pipeline_flow.py — Prefect orchestration DAG for the fraud detection
ETL pipeline (Module 3, Section 2: Implement an Orchestration DAG).

Prefect is used here in place of Airflow: both are accepted by the
assignment, and Prefect's lighter runtime made it practical to actually
execute this flow end-to-end inside a sandboxed development environment,
rather than only writing a DAG file that describes execution without
running it. The task graph and dependency structure (ingest -> clean ->
transform -> integrate -> validate -> save) is the deliverable either
orchestrator would express; only the scheduler syntax differs.

Run with: python dags/fraud_pipeline_flow.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prefect import flow, task
from src.pipeline.etl import ingest, clean, transform, integrate, save
from src.pipeline.validate import run_validation_suite
import logging

logger = logging.getLogger("fraud_pipeline")


@task(name="ingest_data", retries=2, retry_delay_seconds=10)
def ingest_task():
    return ingest()


@task(name="clean_data")
def clean_task(df):
    return clean(df)


@task(name="transform_data")
def transform_task(df):
    return transform(df)


@task(name="integrate_data")
def integrate_task(df):
    return integrate(df)


@task(name="validate_data")
def validate_task(df):
    """Runs the Great Expectations suite; the flow halts if validation fails,
    which is the entire point of validating before, not after, a model
    trains on this data."""
    results = run_validation_suite(df)
    if not results["success"]:
        raise ValueError(f"Data validation FAILED: {results['failed_expectations']}")
    logger.info(f"VALIDATE: all {results['total_expectations']} expectations passed")
    return df


@task(name="save_data")
def save_task(df):
    save(df)
    return df


@flow(name="fraud-detection-etl-pipeline", log_prints=True)
def fraud_pipeline_flow():
    """
    DAG structure:
    ingest -> clean -> transform -> integrate -> validate -> save
    Each task depends on the prior task's output (a linear DAG), matching
    the pipeline stages defined in Module 3 Section 2.
    """
    raw_df = ingest_task()
    cleaned_df = clean_task(raw_df)
    transformed_df = transform_task(cleaned_df)
    integrated_df = integrate_task(transformed_df)
    validated_df = validate_task(integrated_df)
    final_df = save_task(validated_df)
    print(f"Pipeline complete: {len(final_df)} rows ready for modeling (Module 3 -> Module 4 handoff)")
    return final_df


if __name__ == "__main__":
    fraud_pipeline_flow()
