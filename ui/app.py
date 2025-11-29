import streamlit as st
import requests

API_URL = "https://fraud-llm-explainer-production.up.railway.app/predict_and_explain"

st.set_page_config(page_title="Fraud Explainability Agent", layout="wide")

st.title("🔍 Fraud Detection & LLM Explainability Dashboard")

st.markdown("""
A hybrid ML + LLM system that estimates fraud probability using a Random Forest model 
and generates human-readable explanations using an LLM (OpenAI or Gemini).
""")

# -----------------------------
# Correct Feature Names (MUST MATCH TRAINING)
# -----------------------------
FEATURE_NAMES = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
num_features = len(FEATURE_NAMES)

# Input Section
st.subheader("📝 Enter Transaction Features")

values = []
for name in FEATURE_NAMES:
    val = st.number_input(name, value=0.0)
    values.append(val)

if st.button("Analyze Transaction"):

    payload = {
        "features": values,
        "feature_names": FEATURE_NAMES
    }

    with st.spinner("Analyzing..."):
        resp = requests.post(API_URL, json=payload)

    if resp.status_code != 200:
        st.error("API Error: " + resp.text)
    else:
        result = resp.json()

        # Risk Panel
        prob = result["fraud_probability"]
        if prob >= 0.8:
            color = "red"
            label = "⚠️ High Risk"
        elif prob >= 0.4:
            color = "orange"
            label = "🟧 Medium Risk"
        else:
            color = "green"
            label = "🟩 Low Risk"

        st.markdown(
            f"## **Risk Level:** <span style='color:{color}'>{label}</span>",
            unsafe_allow_html=True
        )
        st.metric("Fraud Probability", f"{prob:.2f}")

        # SHAP Contributions
        st.subheader("🔬 SHAP Feature Contributions")
        st.json(result.get("shap_contributions", {}))

        # LLM Explanation
        st.subheader("🧠 LLM Explanation")
        st.write(result.get("llm_explanation", "No explanation available."))
