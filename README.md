This project implements an end-to-end fraud detection and explainability system combining classical ML, modern XAI, and LLM-based reasoning. It predicts fraudulent transactions, explains the decision using SHAP, and generates natural-language risk reports using an LLM agent.

Key Features
1. Fraud Detection Model

Trained on a highly imbalanced dataset (~0.17% fraud)

Applied SMOTE for resampling

Random Forest classifier achieving:

99.9% accuracy

0.86 F1 score

0.96 ROC-AUC

Full preprocessing → scaling → inference pipeline

2. SHAP-Based Explainability

Feature-level contributions for every prediction

Positive vs negative risk factors

Human-interpretable risk summaries

API-ready explanation format

3. LLM Fraud Explainability Agent

A custom-built reasoning agent using Gemini/OpenAI that:

Converts SHAP insights into human-readable explanations

Performs multi-step reasoning

Generates customer-friendly risk narratives

Uses tool-use: SHAP → Summary → LLM → Explanation

4. Production-Ready FastAPI Service

/predict_and_explain endpoint

Returns:

Fraud probability

SHAP values

Risk summary

LLM narrative explanation

Fully documented with /docs (Swagger)

5. MLOps & Deployment

This project is fully containerized and deployment-ready:

FastAPI backend (ML + SHAP + LLM agent)

Redis caching layer

Streamlit UI for interactive fraud analysis

MLflow for experiment tracking

Docker multi-service architecture (backend + UI + Redis)

Railway cloud deployment support

Tech Stack
Layer	Tools
ML	Python, Scikit-learn, SMOTE, Random Forest
Explainability	SHAP, LIME
LLM Agent	Gemini API / OpenAI API
Backend	FastAPI, Pydantic
Caching	Redis
UI	Streamlit
Deployment	Docker, Railway
Experiment Tracking	MLflow
