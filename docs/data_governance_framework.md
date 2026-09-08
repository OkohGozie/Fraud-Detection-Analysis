# Data Governance Framework

Builds directly on the Ethical AI Charter (Module 1) and Data Governance
Principles (Module 2).

## Access Control Matrix

| Role                | Raw Data Access | Processed Data Access | Model/Feature Access | Audit Log Access |
|----------------------|:---------------:|:----------------------:|:---------------------:|:------------------:|
| Data Engineer         | Read/Write       | Read/Write              | Read                   | Read               |
| Data Science Lead     | Read             | Read/Write              | Read/Write              | Read               |
| Fraud Analyst         | No               | No (dashboard only)     | No (scores only, via dashboard) | No                |
| Compliance Officer    | No               | Read (aggregate reports only) | No               | Read               |
| AI & Data Ethics Committee | No          | No                      | Read (fairness audit results only) | Read      |

## Retention Policy
- Raw transaction data: retained per standard banking record-keeping
  requirements and NDPA 2023 guidance, with a documented deletion schedule
  once the retention purpose (fraud investigation window) has lapsed.
- Processed/modeling data: retained for the active model's lifecycle plus
  one full retraining cycle, then archived or deleted per the same schedule.
- Privacy audit logs: retained longer than the underlying data itself
  (audit trails should outlive the data they describe), consistent with
  standard compliance audit practice.

## Change Control
Any change to the expectation suite (`validate.py`), the feature
engineering logic (`etl.py`), or the fairness thresholds requires a pull
request reviewed by the Data Science Lead, with a corresponding entry in
the Technical RAID Log (Module 2) if the change introduces new risk.

## Escalation Path
A failed Great Expectations validation halts the pipeline automatically
(see `dags/fraud_pipeline_flow.py`) and is not overridden without sign-off
from the Data Science Lead, consistent with the human-in-the-loop principle
established in Module 1.
