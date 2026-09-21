"""
explanation_store.py — Explanation Store (Final Project, Section 5)

Every prediction the model makes is stored alongside its explanation, so a
regulator, auditor, or the Ethics Committee can later answer "why did the
model flag this specific transaction on this specific date" without needing
to re-run anything. This is the audit backbone that makes the fairness and
explainability work in Module 4 actually usable after the fact, not just at
development time.

Design choice: append-only JSONL file, mirroring the audit_logging.py pattern
from Module 3 (privacy_audit_log.jsonl). In production this would be a
proper database table with the same schema, but the append-only, tamper-
evident principle stays the same regardless of storage backend.
"""

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path

STORE_PATH = Path("/home/claude/final_project/monitoring/explanation_store.jsonl")


def _record_hash(record: dict) -> str:
    """A simple hash chain: each record's hash covers its own content plus
    the previous record's hash, so any retroactive edit to an old record
    breaks every hash after it. This is the same tamper-evidence principle
    audit logs use, without needing a full blockchain to get it."""
    payload = json.dumps(record, sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()


def store_explanation(transaction_id: str, prediction: dict, explanation: dict, model_version: str = "fraud-detection-v1/2"):
    """Appends one prediction + its explanation to the audit-traceable store."""
    prev_hash = _last_hash()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "transaction_id": transaction_id,
        "model_version": model_version,
        "fraud_probability": prediction.get("fraud_probability"),
        "predicted_class": prediction.get("predicted_class"),
        "priority_tier": prediction.get("priority_tier"),
        "top_contributing_features": explanation.get("top_contributing_features"),
        "explanation_method": explanation.get("method", "SHAP"),
        "prev_hash": prev_hash,
    }
    record["record_hash"] = _record_hash(record)

    with open(STORE_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return record


def _last_hash() -> str:
    if not STORE_PATH.exists():
        return "GENESIS"
    with open(STORE_PATH) as f:
        lines = f.readlines()
    if not lines:
        return "GENESIS"
    return json.loads(lines[-1])["record_hash"]


def verify_chain() -> bool:
    """Walks the whole store and confirms no record was altered after the
    fact. Returns True if the chain is intact."""
    if not STORE_PATH.exists():
        return True
    with open(STORE_PATH) as f:
        lines = [json.loads(l) for l in f]
    expected_prev = "GENESIS"
    for rec in lines:
        if rec["prev_hash"] != expected_prev:
            return False
        check = dict(rec)
        stored_hash = check.pop("record_hash")
        if _record_hash(check) != stored_hash:
            return False
        expected_prev = stored_hash
    return True


def lookup(transaction_id: str):
    """Retrieves the stored explanation for a specific transaction, the
    core audit use case: 'show me why transaction X was flagged.'"""
    if not STORE_PATH.exists():
        return None
    with open(STORE_PATH) as f:
        for line in f:
            rec = json.loads(line)
            if rec["transaction_id"] == transaction_id:
                return rec
    return None


if __name__ == "__main__":
    if STORE_PATH.exists():
        STORE_PATH.unlink()

    demo_cases = [
        ("TXN-2026-001", {"fraud_probability": 0.9999, "predicted_class": "fraud", "priority_tier": "HIGH"},
         {"top_contributing_features": ["V12", "V14", "V10", "hour_of_day"], "method": "SHAP"}),
        ("TXN-2026-002", {"fraud_probability": 0.001, "predicted_class": "legit", "priority_tier": "LOW"},
         {"top_contributing_features": ["Amount", "V5"], "method": "SHAP"}),
        ("TXN-2026-003", {"fraud_probability": 0.15, "predicted_class": "legit", "priority_tier": "LOW"},
         {"top_contributing_features": ["V9", "V22"], "method": "SHAP"}),
    ]
    for txn_id, pred, exp in demo_cases:
        store_explanation(txn_id, pred, exp)

    print(f"Explanation store written to: {STORE_PATH}")
    print(f"Chain integrity check: {'PASS' if verify_chain() else 'FAIL'}")
    print(f"\nLookup demo (TXN-2026-001):")
    print(json.dumps(lookup("TXN-2026-001"), indent=2))

    print(f"\n--- Tamper test ---")
    with open(STORE_PATH) as f:
        lines = f.readlines()
    tampered = json.loads(lines[0])
    tampered["fraud_probability"] = 0.01
    lines[0] = json.dumps(tampered) + "\n"
    with open(STORE_PATH, "w") as f:
        f.writelines(lines)
    print(f"After tampering with record 1: chain integrity = {'PASS' if verify_chain() else 'FAIL (tamper detected, as expected)'}")
