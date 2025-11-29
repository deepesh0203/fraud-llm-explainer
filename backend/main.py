# backend/main.py

import os
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from utils.shap_utils import get_shap_for_instance, build_risk_summary
from utils.llm_agent import LLMAgent

load_dotenv()

# ---------------------------
# Config
# ---------------------------
MODEL_PATH = "model/fraud_model.pkl"
SCALER_PATH = "model/scaler.pkl"
SHAP_EXPLAINER_PATH = "model/shap_explainer.pkl"

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"

# ---------------------------
# Load Artifacts
# ---------------------------
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

try:
    shap_explainer = joblib.load(SHAP_EXPLAINER_PATH)
except:
    shap_explainer = None

llm_agent = LLMAgent(provider=LLM_PROVIDER)

app = FastAPI(title="Fraud Explainability API")

# ---------------------------
# Request Model
# ---------------------------
class Transaction(BaseModel):
    features: list
    feature_names: list

# ---------------------------
# Endpoints
# ---------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_and_explain")
def predict_and_explain(payload: Transaction):

    # Convert to DataFrame
    try:
        df = pd.DataFrame([payload.features], columns=payload.feature_names)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid features: {e}")

    # Scale
    X_scaled = scaler.transform(df)

    # Model prediction
    if hasattr(model, "predict_proba"):
        fraud_prob = float(model.predict_proba(X_scaled)[0][1])
    else:
        fraud_prob = float(model.predict(X_scaled)[0])

    # SHAP
    if shap_explainer:
        shap_vals = get_shap_for_instance(X_scaled, shap_explainer)
    else:
        shap_vals = None

    summary = build_risk_summary(shap_vals, payload.feature_names)
    summary["fraud_probability"] = fraud_prob

    # LLM Explanation
    prompt_inputs = {
        "fraud_probability": fraud_prob,
        "top_positive_factors": summary["top_positive_factors"],
        "top_negative_factors": summary["top_negative_factors"],
        "feature_values": dict(zip(payload.feature_names, payload.features))
    }

    explanation = llm_agent.explain(prompt_inputs)
    summary["llm_explanation"] = explanation

    return summary
