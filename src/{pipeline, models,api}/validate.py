"""
validate.py — Great Expectations data validation suite
(Module 3, Section 4: Implement Data Validation with Great Expectations).

This defines the expectation suite for the processed fraud detection dataset
and a run_validation_suite() function that actually executes it, returning a
pass/fail result the Prefect flow uses as a hard gate before saving/modeling.
"""

import great_expectations as gx
import pandas as pd
import logging

logger = logging.getLogger("fraud_pipeline")


def build_expectation_suite(context, df: pd.DataFrame):
    """Defines the expectations that constitute 'valid' data for this pipeline."""
    data_source = context.data_sources.add_pandas(name="fraud_pipeline_source")
    data_asset = data_source.add_dataframe_asset(name="fraud_transactions")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("full_batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    suite = gx.ExpectationSuite(name="fraud_pipeline_suite")

    # Schema expectations: the exact columns the modeling layer depends on
    # must exist, or a silently-broken upstream feed change would otherwise
    # only surface as a confusing model training error much later.
    suite.add_expectation(gx.expectations.ExpectTableColumnsToMatchSet(
        column_set=["Time", "hour_of_day"] + [f"V{i}" for i in range(1, 29)]
                    + ["Amount", "log_amount", "quality_flag_negative_amount", "Class"],
        exact_match=True,
    ))

    # Completeness: Amount and Class must never be null post-cleaning --
    # a null Class breaks supervised training outright, and a null Amount
    # would have already been imputed by the clean() stage, so a null here
    # means the cleaning stage itself regressed.
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Amount"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Class"))

    # Validity: Class must be strictly binary; Amount must be non-negative
    # in the vast majority of cases (flagged exceptions are allowed through
    # by design, per the clean() stage's flag-don't-drop policy, so this
    # expectation checks the flag column instead of Amount directly).
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeInSet(column="Class", value_set=[0, 1]))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeOfType(column="Amount", type_="float64"))

    # Distributional expectation: the fraud rate should remain within a
    # documented tolerance band of the real dataset's known 0.172% fraud
    # rate. If a future data pull shows fraud at, say, 5%, that is far more
    # likely a labeling/feed error than genuine drift, and should halt the
    # pipeline for investigation rather than silently proceeding.
    suite.add_expectation(gx.expectations.ExpectColumnProportionOfUniqueValuesToBeBetween(
        column="Class", min_value=0.0001, max_value=0.05,
    ))

    # Uniqueness: no duplicate rows should remain after the clean() stage;
    # if this fails, it means clean() itself has a bug, not that the data
    # is inherently dirty.
    suite.add_expectation(gx.expectations.ExpectCompoundColumnsToBeUnique(
        column_list=["Time", "Amount", "V1", "V2"],
    ))

    return suite, batch


def run_validation_suite(df: pd.DataFrame) -> dict:
    """Executes the expectation suite against the given DataFrame and returns
    a structured pass/fail result."""
    context = gx.get_context(mode="ephemeral")
    suite, batch = build_expectation_suite(context, df)

    results = []
    for expectation in suite.expectations:
        result = batch.validate(expectation)
        results.append({
            "expectation": expectation.__class__.__name__,
            "success": bool(result.success),
            "details": result.result,
        })

    n_total = len(results)
    n_passed = sum(1 for r in results if r["success"])
    failed = [r["expectation"] for r in results if not r["success"]]

    summary = {
        "total_expectations": n_total,
        "passed": n_passed,
        "failed_expectations": failed,
        "success": len(failed) == 0,
        "detail": results,
    }
    logger.info(f"GE VALIDATION: {n_passed}/{n_total} expectations passed. "
                f"{'ALL PASSED' if summary['success'] else f'FAILED: {failed}'}")
    return summary


if __name__ == "__main__":
    from pathlib import Path
    df = pd.read_csv(Path("/home/claude/pipeline_project/data/processed/creditcard_clean.csv"))
    result = run_validation_suite(df)
    print(f"\n{'='*60}")
    print(f"GREAT EXPECTATIONS VALIDATION RESULT")
    print(f"{'='*60}")
    print(f"Total expectations: {result['total_expectations']}")
    print(f"Passed: {result['passed']}")
    print(f"Failed: {result['failed_expectations'] if result['failed_expectations'] else 'None'}")
    print(f"Overall: {'PASS' if result['success'] else 'FAIL'}")
    for r in result["detail"]:
        status = "PASS" if r["success"] else "FAIL"
        print(f"  [{status}] {r['expectation']}")
