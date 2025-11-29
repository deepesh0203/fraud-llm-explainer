import os
import json
import logging
from dotenv import load_dotenv
from backend.utils.redis_client import get_cache, set_cache

load_dotenv()
logger = logging.getLogger(__name__)

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")
REDIS_ENABLED = os.getenv("REDIS_ENABLED", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

openai_client = None
gemini_client = None

# Initialize OpenAI
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
        logger.info("OpenAI client initialized.")
    except Exception as e:
        logger.error(f"OpenAI init failed: {e}")

# Initialize Gemini
if GEMINI_API_KEY:
    try:
        from google.genai import Client
        gemini_client = Client(api_key=GEMINI_API_KEY)
        logger.info("Gemini client initialized.")
    except Exception as e:
        logger.error(f"Gemini init failed: {e}")


# ======================================================
#                LLM AGENT CLASS
# ======================================================
class LLMAgent:
    def __init__(self, provider=LLM_PROVIDER):
        self.provider = provider.lower()

    # ------------------------------------------------------
    # PUBLIC CALL
    # ------------------------------------------------------
    def explain(self, inputs: dict) -> str:

        cache_key = "llm:" + str(hash(str(inputs)))
        if REDIS_ENABLED:
            cached = get_cache(cache_key)
            if cached:
                return cached

        prompt = self._build_prompt(inputs)

        # Try OpenAI
        if self.provider == "openai" and openai_client:
            try:
                out = self._call_openai(prompt)
                if REDIS_ENABLED:
                    set_cache(cache_key, out)
                return out
            except Exception:
                logger.exception("OpenAI failed → fallback to Gemini.")

        # Try Gemini
        if gemini_client:
            try:
                out = self._call_gemini(prompt)
                if REDIS_ENABLED:
                    set_cache(cache_key, out)
                return out
            except Exception:
                logger.exception("Gemini failed.")

        # Last fallback
        fallback = self._fallback()
        if REDIS_ENABLED:
            set_cache(cache_key, fallback)
        return fallback

    # ------------------------------------------------------
    # OpenAI call
    # ------------------------------------------------------
    def _call_openai(self, prompt: str) -> str:
        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

        resp = openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a fraud analyst."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=350
        )
        return resp.choices[0].message.content.strip()

    # ------------------------------------------------------
    # Gemini call
    # ------------------------------------------------------
    def _call_gemini(self, prompt: str) -> str:

        response = gemini_client.models.generate_content(
            model="gemini-flash-latest",
            contents=[prompt]
        )

        return response.text.strip()

    # ------------------------------------------------------
    # Prompt Builder
    # ------------------------------------------------------
    def _build_prompt(self, inputs: dict) -> str:
        fp = inputs.get("fraud_probability", 0.0)
        pos = inputs.get("top_positive_factors", {})
        neg = inputs.get("top_negative_factors", {})
        vals = inputs.get("feature_values", {})

        pos_str = "\n".join([f"- {k}: {v:.3f}" for k, v in pos.items()]) or "None"
        neg_str = "\n".join([f"- {k}: {v:.3f}" for k, v in neg.items()]) or "None"
        val_str = "\n".join([f"- {k}: {v}" for k, v in vals.items()]) or "None"

        return f"""
Fraud Probability: {fp:.3f}

Top Positive Indicators:
{pos_str}

Top Negative Indicators:
{neg_str}

Feature Values:
{val_str}

Task:
1. Give a professional fraud analyst explanation.
2. Rewrite it in simple customer-friendly language.
3. Provide recommended next steps.
""".strip()

    # ------------------------------------------------------
    # Fallback
    # ------------------------------------------------------
    def _fallback(self):
        return (
            "LLM unavailable. Basic explanation:\n"
            "The model detected unusual patterns and requires manual review.\n"
            "Customer-friendly: 'We need to verify a recent transaction for safety.'"
        )