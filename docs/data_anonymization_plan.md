# Data Anonymization Plan

## Current State (Modeling Dataset)
The ULB Credit Card Fraud dataset used in this pipeline is already
PCA-anonymized at source: the 28 V-features are principal components
derived from the original transaction features specifically so the
original, potentially identifying features cannot be reverse-engineered
(Kaggle, 2018). Time and Amount are retained in original form because
they carry no direct identifying information on their own.

## Production Anonymization Requirements (beyond this dataset)
A production deployment joining in KYC and device metadata (per the
Module 2 Data Dictionary) would introduce genuine PII requiring active
anonymization, not just inherited anonymization from a public dataset.

| PII Field                | Anonymization Technique | Rationale |
|---------------------------|--------------------------|-----------|
| Customer name              | Remove entirely from model-facing tables; retain only in KYC system of record | Not predictive; pure re-identification risk |
| Account number              | One-way salted hash (SHA-256) | Allows joining/deduplication without reversibility |
| Phone number / email        | One-way salted hash (SHA-256) | Same as above |
| Device ID / IP address      | Truncate IP to /24 subnet; hash device ID | Preserves geographic/device-pattern signal for fraud detection while reducing individual identifiability |
| Date of birth                | Generalize to age band (18-25, 26-35, etc.) | Age band retains predictive value without exact DOB |

## Guiding Principle
Anonymization is applied at the earliest pipeline stage possible (ingestion),
not deferred to the modeling stage, so that no raw PII ever reaches the
feature engineering, model training, or dashboard layers, consistent with
data minimization under NDPA 2023 Section 24 and Article 5 of GDPR (for the
EU-sourced original dataset's underlying data subjects).

## Reference
Kaggle / Machine Learning Group, Universite Libre de Bruxelles. (2018).
Credit card fraud detection [Data set].
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
