"""
audit_logging.py — Privacy Audit Logging
(Module 3, Section 6: Implement Privacy Audit Logging for all data access
and transformations).

A decorator-based approach so every pipeline function that touches data is
automatically logged, without each function needing to remember to log
itself, which is how audit logging gets silently skipped in real systems.
"""

import functools
import logging
import json
from datetime import datetime, timezone
from pathlib import Path

AUDIT_LOG_PATH = Path("/home/claude/pipeline_project/docs/privacy_audit_log.jsonl")

audit_logger = logging.getLogger("privacy_audit")
audit_logger.setLevel(logging.INFO)


def audit_log(action: str, user: str = "pipeline_service_account"):
    """Decorator: wraps a data-touching function and writes a structured
    audit record every time it runs, capturing who/what/when, consistent
    with the Data Governance Framework's access-control requirement."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            timestamp = datetime.now(timezone.utc).isoformat()
            record = {
                "timestamp": timestamp,
                "user": user,
                "action": action,
                "function": func.__name__,
            }
            try:
                result = func(*args, **kwargs)
                record["status"] = "success"
                if hasattr(result, "shape"):
                    record["rows_affected"] = int(result.shape[0])
                return result
            except Exception as e:
                record["status"] = "failure"
                record["error"] = str(e)
                raise
            finally:
                with open(AUDIT_LOG_PATH, "a") as f:
                    f.write(json.dumps(record) + "\n")
        return wrapper
    return decorator


if __name__ == "__main__":
    # Demonstration run: wrap the real pipeline stages and show the audit
    # trail this produces.
    import sys
    sys.path.append("/home/claude/pipeline_project")
    from src.pipeline.etl import ingest, clean, transform, integrate

    audited_ingest = audit_log("READ_RAW_DATA")(ingest)
    audited_clean = audit_log("TRANSFORM_CLEAN")(clean)
    audited_transform = audit_log("TRANSFORM_FEATURE_ENGINEER")(transform)
    audited_integrate = audit_log("TRANSFORM_SCHEMA_ALIGN")(integrate)

    if AUDIT_LOG_PATH.exists():
        AUDIT_LOG_PATH.unlink()

    df = audited_ingest()
    df = audited_clean(df)
    df = audited_transform(df)
    df = audited_integrate(df)

    print("Privacy audit log written to:", AUDIT_LOG_PATH)
    print("\nAudit trail contents:")
    with open(AUDIT_LOG_PATH) as f:
        for line in f:
            entry = json.loads(line)
            print(f"  [{entry['timestamp']}] {entry['user']} | {entry['action']} "
                  f"| {entry['function']} | {entry['status']} "
                  f"| rows={entry.get('rows_affected', 'n/a')}")
