import shap
import numpy as np
import pandas as pd

# Human-readable names for the demo
FEATURE_LABELS = {
    'education_num':    'Education level',
    'age':              'Candidate age',
    'hours_per_week':   'Weekly hours',
    'capital_gain':     'Financial capital',
    'capital_loss':     'Financial losses',
}

def get_shap_explainer(model, X_train):
    clf     = model.named_steps['clf']
    scaler  = model.named_steps['scaler']
    X_scaled = scaler.transform(X_train)
    explainer = shap.TreeExplainer(clf)
    return explainer, scaler

def explain_single(explainer, scaler, X_row):
    X_scaled  = scaler.transform(X_row)
    shap_vals = explainer.shap_values(X_scaled)[0]
    return dict(zip(X_row.columns, shap_vals.tolist()))

def top_factors(shap_dict, n=5):
    labeled = {
        FEATURE_LABELS.get(k, k): v for k, v in shap_dict.items()
    }
    return sorted(labeled.items(), key=lambda x: abs(x[1]), reverse=True)[:n]