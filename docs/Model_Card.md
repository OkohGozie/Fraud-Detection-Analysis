# Model Card: Fraud Detection Classifier

Following the standard structure proposed in Mitchell et al. (2019),
*Model Cards for Model Reporting*.

## Model Details
- **Model type:** Logistic Regression (class-weighted, `class_weight="balanced"`)
- **Selected from:** 3 candidate algorithms (Logistic Regression, Random Forest, XGBoost) plus a majority-class baseline
- **Selection criterion:** Highest F1 score on held-out test set
- **Version:** 1.0 (Module 4)
- **Training framework:** scikit-learn 1.4, tracked via MLflow
- **License / owner:** Academic capstone project (Nexford University, BAN6800)

## Intended Use
- **Primary intended use:** Flag individual retail banking transactions with an elevated probability of fraud, for review by a human fraud analyst.
- **Primary intended users:** Fraud analysts, compliance officers (aggregate reporting only).
- **Out-of-scope uses:** This model is NOT intended to automatically block, freeze, or reverse transactions. No account action should be taken on this model's output alone (human-in-the-loop principle, Module 1 Ethical AI Vision).

## Factors
- **Relevant factors:** Transaction amount, time of day, and 28 anonymized behavioral components (V1-V28).
- **Evaluation factors:** Performance was evaluated overall and via a proxy grouping (transaction amount band), due to the absence of demographic fields in the source dataset (see Ethical Considerations below).

## Metrics
- Accuracy, Precision, Recall, F1, ROC-AUC (see Performance Evaluation in the main report)
- Model selection used F1 rather than accuracy, since accuracy is misleading under this dataset's severe class imbalance (a majority-class baseline achieves 99.84% accuracy while catching zero fraud cases).

## Evaluation Data
- 25% stratified hold-out split of the Module 3 processed dataset (5,000 rows, 8 fraud cases).
- **Limitation:** this proxy dataset (20,000 rows total) is a fraction of the real ULB Credit Card Fraud dataset's 284,807 rows; the small absolute number of fraud cases in the test set (8) limits the statistical precision of all reported metrics. Results should be re-validated on the full real dataset before any production use.

## Training Data
- 75% stratified split of the same processed dataset (15,000 rows, 26 fraud cases).
- Source: Credit Card Fraud Detection dataset proxy (see Module 3 methodology note); PCA-anonymized at source.

## Quantitative Analyses
- See Performance Evaluation and Fairness Analysis sections of the main Model Validation Report for full breakdowns, including per-amount-band selection rates, true/false positive rates, and the disparate impact ratio.

## Ethical Considerations
- **Data:** The evaluation dataset contains no demographic fields (age, gender, region), because the original dataset is PCA-anonymized specifically to prevent re-identification. This is a genuine ethical safeguard for the original data subjects, but it also means true demographic fairness auditing (Demographic Parity, Equalized Odds across protected groups) could not be performed on this dataset. A proxy analysis using transaction amount band was performed instead and is clearly labeled as a proxy, not a substitute, throughout this report.
- **Human life / high-stakes impact:** A false positive can inconvenience a legitimate customer (a flagged transaction under review); a false negative allows fraud to proceed undetected. Given this asymmetry, the model is deployed strictly as a decision-support signal, never as an automated blocker.
- **Mitigations:** Fairlearn's ThresholdOptimizer was applied when the proxy disparate impact ratio fell below the 0.80 four-fifths-rule threshold, reducing the demographic parity difference from 0.0136 to 0.0012 in testing.

## Caveats and Recommendations
- Near-perfect performance metrics (ROC-AUC approximately 0.9997-1.0) on this proxy dataset are almost certainly optimistic, both because the synthetic proxy's fraud class was constructed with a pronounced statistical separation on a handful of features, and because of the small absolute test-set size (8 fraud cases). Performance should be re-validated on the full real dataset (284,807 rows, 492 fraud cases) before any deployment decision.
- True demographic fairness auditing is a prerequisite for production deployment, not an optional enhancement, and requires production KYC data under NDPA-governed access per the Module 2 Data Privacy Plan.
- This Model Card should be re-issued after retraining on real data, since metric values, fairness results, and top SHAP features may all shift materially.

## Reference
Mitchell, M., Wu, S., Zaldivar, A., Barnes, P., Vasserman, L., Hutchinson, B., Spitzer, E., Raji, I. D., & Gebru, T. (2019). Model cards for model reporting. *Proceedings of the Conference on Fairness, Accountability, and Transparency*, 220-229. https://doi.org/10.1145/3287560.3287596
