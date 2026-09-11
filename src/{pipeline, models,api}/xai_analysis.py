"""
xai_analysis.py — Module 4 Section 6: Explainable AI
Global feature importance (SHAP), local explanations (SHAP + LIME), and
counterfactual explanations (DiCE), run against the actual best model
trained in train_models.py.
"""

import joblib
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from pathlib import Path

ARTIFACT_DIR = Path("/home/claude/model_dev/artifacts")

model = joblib.load(ARTIFACT_DIR / "best_model.joblib")
feature_cols = joblib.load(ARTIFACT_DIR / "feature_cols.joblib")
X_test = joblib.load(ARTIFACT_DIR / "X_test.joblib")
y_test = joblib.load(ARTIFACT_DIR / "y_test.joblib")
X_train = joblib.load(ARTIFACT_DIR / "X_train.joblib")

print(f"Loaded model: {type(model).__name__}")
print(f"Test set: {len(X_test)} rows, {y_test.sum()} fraud cases")

# ---------- SHAP: Global feature importance ----------
# LinearExplainer is used because the selected best model (LogisticRegression)
# is a linear model; SHAP's model-specific explainers are exact and far
# faster than the general KernelExplainer for this model class.
print("\nComputing SHAP values (global)...")
background = shap.sample(X_train, 100, random_state=42)
explainer = shap.LinearExplainer(model, background)
shap_values = explainer.shap_values(X_test)
shap_values = np.asarray(shap_values, dtype=np.float64)
X_test_values = X_test.to_numpy(dtype=np.float64)

plt.figure()
shap.summary_plot(shap_values, X_test_values, feature_names=feature_cols, show=False, max_display=12)
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "shap_summary_global.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved: shap_summary_global.png")

# Global feature importance ranking (mean absolute SHAP value)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
importance_df = pd.DataFrame({"feature": feature_cols, "mean_abs_shap": mean_abs_shap})
importance_df = importance_df.sort_values("mean_abs_shap", ascending=False)
importance_df.to_csv(ARTIFACT_DIR / "shap_feature_importance.csv", index=False)
print("\nTop 10 features by global SHAP importance:")
print(importance_df.head(10).to_string(index=False))

# ---------- SHAP: Local explanation for one flagged transaction ----------
fraud_idx_in_test = np.where(y_test.values == 1)[0]
if len(fraud_idx_in_test) > 0:
    local_idx = fraud_idx_in_test[0]
    print(f"\nLocal SHAP explanation for test-set fraud case (index {local_idx}):")
    local_shap = shap_values[local_idx]
    local_features = X_test.iloc[local_idx]
    top_contribs = pd.DataFrame({
        "feature": feature_cols,
        "value": local_features.values,
        "shap_contribution": local_shap,
    }).sort_values("shap_contribution", key=abs, ascending=False)
    print(top_contribs.head(8).to_string(index=False))
    top_contribs.to_csv(ARTIFACT_DIR / "shap_local_explanation_sample.csv", index=False)

    plt.figure()
    shap.force_plot(explainer.expected_value, local_shap, local_features,
                     feature_names=feature_cols, matplotlib=True, show=False)
    plt.tight_layout()
    plt.savefig(ARTIFACT_DIR / "shap_local_force_plot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: shap_local_force_plot.png")

# ---------- LIME: Local explanation (independent method, cross-check vs SHAP) ----------
print("\nComputing LIME explanation for the same case...")
from lime.lime_tabular import LimeTabularExplainer

lime_explainer = LimeTabularExplainer(
    X_train.values, feature_names=feature_cols, class_names=["Legit", "Fraud"],
    mode="classification", random_state=42,
)
lime_exp = lime_explainer.explain_instance(
    X_test.iloc[local_idx].values, model.predict_proba, num_features=8,
)
lime_list = lime_exp.as_list()
print("LIME top contributing features:")
for feat, weight in lime_list:
    print(f"  {feat}: {weight:.4f}")

lime_df = pd.DataFrame(lime_list, columns=["feature_condition", "weight"])
lime_df.to_csv(ARTIFACT_DIR / "lime_local_explanation_sample.csv", index=False)

fig = lime_exp.as_pyplot_figure()
plt.tight_layout()
plt.savefig(ARTIFACT_DIR / "lime_local_explanation.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: lime_local_explanation.png")

# ---------- DiCE: Counterfactual explanation ----------
print("\nComputing counterfactual explanation (DiCE)...")
import dice_ml

full_train_df = X_train.copy()
y_train_full = pd.read_csv("/home/claude/model_dev/data_creditcard_clean.csv").loc[X_train.index, "Class"]

dice_data = X_train.copy()
dice_data["quality_flag_negative_amount"] = dice_data["quality_flag_negative_amount"].astype(int)
dice_data["Class"] = model.predict(X_train).astype(int)  # use model's own training predictions as the outcome column DiCE needs

query_instance = X_test.iloc[[local_idx]].copy()
query_instance["quality_flag_negative_amount"] = query_instance["quality_flag_negative_amount"].astype(int)

d = dice_ml.Data(dataframe=dice_data, continuous_features=[c for c in feature_cols if c != "quality_flag_negative_amount"], outcome_name="Class")
m = dice_ml.Model(model=model, backend="sklearn")
dice_exp = dice_ml.Dice(d, m, method="random")

try:
    cf = dice_exp.generate_counterfactuals(query_instance, total_CFs=3, desired_class=0, random_seed=42)
    cf_df = cf.cf_examples_list[0].final_cfs_df
    cf_df.to_csv(ARTIFACT_DIR / "dice_counterfactuals_sample.csv", index=False)
    print("Counterfactuals (what would need to change to flip this prediction to 'legit'):")
    print(cf_df[[c for c in cf_df.columns if c in ["Amount", "log_amount", "hour_of_day", "Class"]]].to_string(index=False))
except Exception as e:
    print(f"DiCE counterfactual generation note: {e}")
    print("Falling back to manual counterfactual search for transparency:")
    # Manual counterfactual: perturb the top-3 SHAP-driving features toward
    # the legitimate-class mean until the model's predicted class flips.
    manual_cf = X_test.iloc[[local_idx]].copy()
    top3_features = importance_df.head(3)["feature"].tolist()
    legit_means = X_train[y_train_full == 0][top3_features].mean() if y_train_full is not None else X_train[top3_features].mean()
    for feat in top3_features:
        manual_cf[feat] = legit_means[feat]
    new_pred = model.predict(manual_cf)[0]
    new_proba = model.predict_proba(manual_cf)[0][1]
    print(f"  Original prediction: Fraud (probability={model.predict_proba(X_test.iloc[[local_idx]])[0][1]:.4f})")
    print(f"  After setting {top3_features} to their legitimate-class average values:")
    print(f"  New prediction: {'Fraud' if new_pred == 1 else 'Legit'} (probability={new_proba:.4f})")
    manual_cf.to_csv(ARTIFACT_DIR / "manual_counterfactual_sample.csv", index=False)

print("\nXAI analysis complete. All artifacts saved to:", ARTIFACT_DIR)
