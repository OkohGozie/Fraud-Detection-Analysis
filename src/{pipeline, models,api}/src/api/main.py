"""
main.py — FastAPI application serving the fraud detection model.
Run with: uvicorn api.main:app --reload --port 8000
Then POST to http://localhost:8000/predict
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"

app = FastAPI(
    title="Fraud Detection API",
    description="Serves the Module 4 fraud-probability model (LogisticRegression, "
                 "class-weighted) trained on the ULB Credit Card Fraud proxy dataset. "
                 "Human-in-the-loop: this API returns a probability and explanation, "
                 "not an automated account action, per the Ethical AI Vision (Module 1).",
    version="1.0.0",
)

model = joblib.load(ARTIFACT_DIR / "best_model.joblib")
feature_cols = joblib.load(ARTIFACT_DIR / "feature_cols.joblib")
importance_df = pd.read_csv(ARTIFACT_DIR / "shap_feature_importance.csv")
TOP_FEATURES = importance_df.head(5)["feature"].tolist()


class Transaction(BaseModel):
    # V1-V28 anonymized PCA features; Amount and engineered features.
    # Example values below reflect a typical legitimate transaction.
    V1: float = 0.0; V2: float = 0.0; V3: float = 0.0; V4: float = 0.0
    V5: float = 0.0; V6: float = 0.0; V7: float = 0.0; V8: float = 0.0
    V9: float = 0.0; V10: float = 0.0; V11: float = 0.0; V12: float = 0.0
    V13: float = 0.0; V14: float = 0.0; V15: float = 0.0; V16: float = 0.0
    V17: float = 0.0; V18: float = 0.0; V19: float = 0.0; V20: float = 0.0
    V21: float = 0.0; V22: float = 0.0; V23: float = 0.0; V24: float = 0.0
    V25: float = 0.0; V26: float = 0.0; V27: float = 0.0; V28: float = 0.0
    Amount: float = Field(..., description="Transaction amount", ge=0)
    hour_of_day: int = Field(..., description="Hour of transaction (0-23)", ge=0, le=23)
    quality_flag_negative_amount: bool = False


class PredictionResponse(BaseModel):
    fraud_probability: float
    predicted_class: str
    priority_tier: str
    top_contributing_features: list
    note: str


@app.get("/")
def root():
    return {"service": "Fraud Detection API", "status": "running", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction):
    try:
        data = transaction.dict()
        data["log_amount"] = float(np.log1p(data["Amount"]))
        data["quality_flag_negative_amount"] = int(data["quality_flag_negative_amount"])

        row = pd.DataFrame([{col: data[col] for col in feature_cols}])
        proba = float(model.predict_proba(row)[0][1])
        pred_class = "fraud" if proba >= 0.5 else "legit"

        # Prescriptive priority tier (Module 2, Section 3): weight by loss
        # exposure, not probability alone.
        loss_exposure = proba * data["Amount"]
        if loss_exposure > 500:
            tier = "HIGH — investigate immediately"
        elif loss_exposure > 50:
            tier = "MEDIUM — review within shift"
        else:
            tier = "LOW — routine queue"

        return PredictionResponse(
            fraud_probability=round(proba, 4),
            predicted_class=pred_class,
            priority_tier=tier,
            top_contributing_features=TOP_FEATURES,
            note="Prediction only. No automated account action is taken — per the "
                 "human-in-the-loop principle in the Module 1 Ethical AI Vision, "
                 "a fraud analyst must review before any action is taken.",
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
