# backend/utils/shap_utils.py
import numpy as np
import pandas as pd
from typing import Optional, List, Dict

def get_shap_for_instance(x_instance, shap_explainer):
    """
    Compute SHAP values on demand for a single instance.
    Works for TreeExplainer with binary classification (returns SHAP for positive class).
    """
    if shap_explainer is None:
        # fallback: no explainer
        return np.zeros(x_instance.shape[1])

    shap_vals = shap_explainer.shap_values(x_instance)

    # Situations to handle:
    # 1. shap_vals is a list (binary/multiclass)
    # 2. shap_vals is an ndarray of shape (1, n_features)
    # 3. shap_vals is ndarray shape (n_samples, n_features, n_classes)

    if isinstance(shap_vals, list):
        # positive class = index 1
        shap_arr = np.array(shap_vals[1]).reshape(-1)
    else:
        shap_vals = np.array(shap_vals)
        if shap_vals.ndim == 3:
            # shape (n_samples, n_features, n_classes)
            shap_arr = shap_vals[0, :, 1]
        else:
            # shape (1, n_features)
            shap_arr = shap_vals.reshape(-1)

    return shap_arr

def build_risk_summary(shap_array: np.ndarray, feature_names: Optional[List[str]] = None, top_k: int = 5) -> Dict:
    """
    Convert shap_array (n_features,) -> risk summary JSON.
    Returns:
      {
        "top_positive_factors": {"V14": 0.55, ...},
        "top_negative_factors": {"V3": -0.21, ...},
        "shap_contributions": {"V1": 0.1, ...},
        "risk_score": float
      }
    """
    n = shap_array.shape[0]
    if feature_names is None:
        feature_names = [f"f{i}" for i in range(n)]
    # Pair and sort
    pairs = list(zip(feature_names, shap_array.tolist()))
    sorted_by_abs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
    top = sorted_by_abs[:top_k]
    positives = {k: float(v) for k, v in top if v > 0}
    negatives = {k: float(v) for k, v in top if v < 0}
    shap_contribs = {k: float(v) for k, v in pairs}
    # risk_score: normalized positive contribution vs total abs
    pos_sum = sum([abs(v) for v in shap_contribs.values() if v > 0])
    total_abs = sum([abs(v) for v in shap_contribs.values()]) + 1e-9
    risk_score = (pos_sum / total_abs)
    return {
        "top_positive_factors": positives,
        "top_negative_factors": negatives,
        "shap_contributions": shap_contribs,
        "risk_score": float(risk_score)
    }
