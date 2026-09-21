"""
train_models.py — Module 4: Model Development & Validation
Trains a baseline, Logistic Regression, and Random Forest / XGBoost, all
logged to MLflow. Handles severe class imbalance via class weighting.
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                              roc_auc_score, confusion_matrix, roc_curve, classification_report)
import xgboost as xgb
import matplotlib.pyplot as plt

DATA_PATH = Path("/home/claude/model_dev/data_creditcard_clean.csv")
ARTIFACT_DIR = Path("/home/claude/model_dev/artifacts")
ARTIFACT_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri("sqlite:////home/claude/model_dev/mlruns/mlflow.db")
mlflow.set_experiment("fraud-detection-module4")

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(DATA_PATH)
    # Target and feature separation. Leakage check: 'quality_flag_negative_amount'
    # is retained as a legitimate feature (known at scoring time, not derived
    # from the target), while 'Time' is dropped as a raw feature since it is
    # an artifact of dataset construction (seconds since first transaction in
    # the batch), not a genuine recurring signal in production; hour_of_day,
    # its engineered derivative, is retained instead.
    feature_cols = [c for c in df.columns if c not in ["Class", "Time"]]
    X = df[feature_cols]
    y = df["Class"]
    return X, y, feature_cols


def stratified_split(X, y):
    # Stratified hold-out is used rather than a random split because the
    # target is severely imbalanced (0.17% positive class); a non-stratified
    # split risks a test set with zero or near-zero fraud cases, making
    # evaluation meaningless.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, stratify=y, random_state=RANDOM_STATE
    )
    return X_train, X_test, y_train, y_test


def evaluate(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"]); ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=14, fontweight="bold")
    plt.tight_layout()
    cm_path = ARTIFACT_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(cm_path, dpi=150)
    plt.close()

    # ROC curve plot
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, color="#C41230", linewidth=2, label=f"{model_name} (AUC={metrics['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend()
    plt.tight_layout()
    roc_path = ARTIFACT_DIR / f"roc_curve_{model_name}.png"
    plt.savefig(roc_path, dpi=150)
    plt.close()

    return metrics, y_pred, y_proba, str(cm_path), str(roc_path)


def run_baseline(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="baseline_majority_class"):
        model = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
        model.fit(X_train, y_train)
        # DummyClassifier with most_frequent has no meaningful predict_proba for AUC;
        # assign 0 probability to all for a defined (worst-case) AUC baseline.
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": 0.5,  # undefined/random-chance baseline, documented as such
        }
        mlflow.log_param("model_type", "DummyClassifier_majority_class")
        mlflow.log_metrics(metrics)
        print(f"BASELINE (majority class): {metrics}")
        return metrics


def run_logistic_regression(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="logistic_regression"):
        param_grid = {"C": [0.01, 0.1, 1.0, 10.0]}
        base_model = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(base_model, param_grid, scoring="average_precision", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        metrics, y_pred, y_proba, cm_path, roc_path = evaluate(best_model, X_test, y_test, "LogisticRegression")

        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"LOGISTIC REGRESSION: best_params={grid.best_params_}, metrics={metrics}")
        return best_model, metrics


def run_random_forest(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="random_forest"):
        param_grid = {"n_estimators": [100, 200], "max_depth": [8, 12, None]}
        base_model = RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(base_model, param_grid, scoring="average_precision", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        metrics, y_pred, y_proba, cm_path, roc_path = evaluate(best_model, X_test, y_test, "RandomForest")

        mlflow.log_param("model_type", "RandomForestClassifier")
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"RANDOM FOREST: best_params={grid.best_params_}, metrics={metrics}")
        return best_model, metrics


def run_xgboost(X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name="xgboost"):
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        param_grid = {"max_depth": [4, 6], "learning_rate": [0.05, 0.1], "n_estimators": [150, 250]}
        base_model = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight, eval_metric="aucpr",
            random_state=RANDOM_STATE, n_jobs=-1,
        )
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
        grid = GridSearchCV(base_model, param_grid, scoring="average_precision", cv=cv, n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_

        metrics, y_pred, y_proba, cm_path, roc_path = evaluate(best_model, X_test, y_test, "XGBoost")

        mlflow.log_param("model_type", "XGBClassifier")
        mlflow.log_param("scale_pos_weight", scale_pos_weight)
        mlflow.log_params(grid.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.xgboost.log_model(best_model, "model")

        print(f"XGBOOST: best_params={grid.best_params_}, scale_pos_weight={scale_pos_weight:.1f}, metrics={metrics}")
        return best_model, metrics


if __name__ == "__main__":
    X, y, feature_cols = load_data()
    X_train, X_test, y_train, y_test = stratified_split(X, y)
    print(f"Train: {len(X_train)} rows ({y_train.sum()} fraud) | Test: {len(X_test)} rows ({y_test.sum()} fraud)")

    baseline_metrics = run_baseline(X_train, X_test, y_train, y_test)
    lr_model, lr_metrics = run_logistic_regression(X_train, X_test, y_train, y_test)
    rf_model, rf_metrics = run_random_forest(X_train, X_test, y_train, y_test)
    xgb_model, xgb_metrics = run_xgboost(X_train, X_test, y_train, y_test)

    # Select best model by F1 (balances precision/recall, appropriate given
    # the business need to avoid both missed fraud and excessive false alarms)
    all_results = {
        "LogisticRegression": (lr_model, lr_metrics),
        "RandomForest": (rf_model, rf_metrics),
        "XGBoost": (xgb_model, xgb_metrics),
    }
    best_name = max(all_results, key=lambda k: all_results[k][1]["f1"])
    best_model, best_metrics = all_results[best_name]
    print(f"\nBEST MODEL: {best_name} | {best_metrics}")

    joblib.dump(best_model, ARTIFACT_DIR / "best_model.joblib")
    joblib.dump(feature_cols, ARTIFACT_DIR / "feature_cols.joblib")
    joblib.dump(X_test, ARTIFACT_DIR / "X_test.joblib")
    joblib.dump(y_test, ARTIFACT_DIR / "y_test.joblib")
    joblib.dump(X_train, ARTIFACT_DIR / "X_train.joblib")

    with open(ARTIFACT_DIR / "model_selection_summary.txt", "w") as f:
        f.write(f"Baseline (majority class): {baseline_metrics}\n")
        f.write(f"Logistic Regression: {lr_metrics}\n")
        f.write(f"Random Forest: {rf_metrics}\n")
        f.write(f"XGBoost: {xgb_metrics}\n")
        f.write(f"\nBest model selected: {best_name} (highest F1)\n")

    print(f"\nBest model saved to {ARTIFACT_DIR / 'best_model.joblib'}")
