import shap
import numpy as np
import pandas as pd

FEATURE_LABELS = {
    'Education_Num':  'Education level',
    'Age':            'Candidate age',
    'Hours_per_week': 'Weekly hours',
    'Capital_Gain':   'Financial capital',
    'Capital_Loss':   'Financial losses',
}

def get_shap_explainer(model, X_train):
    """TreeExplainer works on raw data — bypass the scaler."""
    clf = model.named_steps['clf']
    # Use a background sample of 200 rows for speed
    background = shap.sample(X_train, 200, random_state=42)
    explainer = shap.TreeExplainer(clf, background)
    return explainer, model.named_steps['scaler']

def explain_single(explainer, scaler, X_row):
    """Pass scaled data to match what the tree actually saw during training."""
    X_scaled = pd.DataFrame(
        scaler.transform(X_row),
        columns=X_row.columns
    )
    shap_vals = explainer.shap_values(X_scaled)
    # GradientBoosting returns a single array, not a list
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]
    return dict(zip(X_row.columns, shap_vals[0].tolist()))

def top_factors(shap_dict, n=5):
    labeled = {FEATURE_LABELS.get(k, k): v for k, v in shap_dict.items()}
    return sorted(labeled.items(), key=lambda x: abs(x[1]), reverse=True)[:n]