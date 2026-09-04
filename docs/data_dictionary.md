# Data Dictionary — Fraud Detection Analytics

Source: Credit Card Fraud Detection dataset (Machine Learning Group, Université
Libre de Bruxelles), via Kaggle. Real, anonymized European card transactions,
September 2013. 284,807 records; 492 confirmed frauds (0.172%).

| Field   | Type    | Description                                                                 | Notes |
|---------|---------|-------------------------------------------------------------------------------|-------|
| Time    | Numeric | Seconds elapsed between this transaction and the first transaction in the dataset | Not PCA-transformed; usable for time-based feature engineering (e.g., transaction velocity) |
| V1–V28  | Numeric | 28 principal components derived from the original transaction features via PCA | Original features withheld for confidentiality (Kaggle, 2018); precise business meaning of each component is not disclosed by the source |
| Amount  | Numeric | Transaction amount                                                            | Not PCA-transformed; a known strong signal in fraud detection literature |
| Class   | Binary  | Target label: 1 = fraudulent transaction, 0 = legitimate transaction          | Severely imbalanced (0.172% positive class) |

## Planned Production Data Sources (beyond the Module 1/2 modeling dataset)

For a production deployment at a real institution, the pipeline in Section 3 of
the accompanying report would additionally ingest:

| Source                        | Type              | Purpose in Pipeline |
|--------------------------------|-------------------|----------------------|
| Core banking transaction feed  | Structured, streaming | Primary scoring input (real-time transaction records) |
| KYC / account metadata         | Structured, batch | Account tenure, verification status — contextual features |
| Device / session metadata      | Structured, semi-structured | Device fingerprint, IP geolocation — anomaly signals |
| Prior fraud case labels        | Structured, batch | Historical ground truth for supervised retraining |

## Data Sensitivity Classification

| Field Category            | Sensitivity | Handling Requirement |
|----------------------------|-------------|------------------------|
| PCA-transformed features (V1–V28) | Low (de-identified) | Standard access control |
| Transaction Amount / Time  | Low | Standard access control |
| KYC / account metadata (production) | High (personal data under NDPA 2023) | Role-based access, encryption at rest, DPIA required |
| Device / session metadata (production) | Medium–High | Encryption in transit, retention limits |
