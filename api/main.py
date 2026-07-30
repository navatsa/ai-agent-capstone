"""
FastAPI service exposing the capstone's SLA-prediction model
(Logistic Regression, class_weight="balanced") for real-time scoring.

Run locally:
    cd capstone/api
    uvicorn main:app --reload --port 8000

Then open http://127.0.0.1:8000/docs for interactive Swagger UI.
"""
from pathlib import Path
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "sla_met_logreg_balanced.joblib"

app = FastAPI(
    title="SLA Prediction API",
    description=(
        "Predicts whether a support case will meet its SLA (`sla_met`), "
        "using the Logistic Regression (class_weight='balanced') model "
        "trained in eda_report.ipynb (F1 ~0.773 on held-out test data)."
    ),
    version="1.0.0",
)

model = None  # populated on startup


@app.on_event("startup")
def load_model() -> None:
    global model
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run the model-persistence cells in notebooks/eda_report.ipynb first."
        )
    model = joblib.load(MODEL_PATH)


class CaseFeatures(BaseModel):
    """Input schema mirrors the feature_cols used to train the model."""

    severity: Literal["Sev-A", "Sev-B", "Sev-C", "Sev-D"] = Field(..., description="Case severity level")
    product_area: str = Field(..., description="e.g. Networking, Identity, Compute, Security")
    issue_type: str = Field(..., description="e.g. Bug, Configuration, Performance, Feature Gap, Outage, How-To")
    complexity: Literal["Low", "Medium", "High"]
    effort_level: Literal["Low", "Medium", "High"]
    has_agent: bool = Field(..., description="Whether an AI/automation agent was involved in the case")
    sla_target_days: int = Field(..., gt=0, description="SLA target window, in days")
    engineer_workload: int = Field(..., ge=0, description="Engineer's concurrent workload at case open")
    queue_depth: int = Field(..., ge=0, description="Queue depth at case open")
    kb_coverage_score: float = Field(..., ge=0, le=1, description="Knowledge-base coverage quality score (0-1)")
    num_actions: int = Field(..., ge=0, description="Number of actions taken on the case")

    class Config:
        json_schema_extra = {
            "example": {
                "severity": "Sev-C",
                "product_area": "Networking",
                "issue_type": "Configuration",
                "complexity": "Medium",
                "effort_level": "High",
                "has_agent": True,
                "sla_target_days": 12,
                "engineer_workload": 20,
                "queue_depth": 44,
                "kb_coverage_score": 0.89,
                "num_actions": 8,
            }
        }


class PredictionResponse(BaseModel):
    sla_met_prediction: bool
    sla_met_probability: float
    sla_missed_probability: float


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: CaseFeatures) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    input_df = pd.DataFrame([features.model_dump()])

    try:
        prediction = bool(model.predict(input_df)[0])
        probabilities = model.predict_proba(input_df)[0]
    except Exception as exc:  # surface pipeline/encoding errors clearly to the caller
        raise HTTPException(status_code=400, detail=f"Prediction failed: {exc}") from exc

    # predict_proba columns follow the classifier's `classes_` order: [0, 1] -> [Missed, Met]
    prob_missed, prob_met = probabilities[0], probabilities[1]

    return PredictionResponse(
        sla_met_prediction=prediction,
        sla_met_probability=round(float(prob_met), 4),
        sla_missed_probability=round(float(prob_missed), 4),
    )
