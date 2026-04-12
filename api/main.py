from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
import numpy as np
import pandas as pd
import pickle, os, sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.schemas      import PredictRequest, AuditDecision
from src.data_loader  import load_as_dataframe
from src.model        import train_baseline
from src.bias_mitigator import train_reweighed
from src.explainer    import get_shap_explainer, explain_single, top_factors
from src.bias_detector import compute_fairness_metrics

MODELS = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    print("Loading models...")
    X_train, X_test, y_train, y_test, s_train, s_test = load_as_dataframe()
    MODELS['baseline']  = train_baseline(X_train, y_train)
    MODELS['mitigated'] = train_reweighed(X_train, y_train, s_train)
    MODELS['explainer'], MODELS['scaler'] = get_shap_explainer(
        MODELS['baseline'], X_train
    )
    MODELS['feature_names'] = list(X_train.columns)
    print("Models ready.")
    yield

app = FastAPI(
    title="AI Decision Auditing API",
    description="Bias-aware ML decisions with fairness metrics and SHAP explanations",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}

@app.post("/predict", response_model=AuditDecision)
def predict(req: PredictRequest):
    try:
        feature_names = MODELS['feature_names']
        row_data = {f: req.features.get(f, 0.0) for f in feature_names}
        X_row = pd.DataFrame([row_data])

        model = MODELS['mitigated'] if req.use_mitigated else MODELS['baseline']

        pred  = model.predict(X_row)[0]
        proba = model.predict_proba(X_row)[0][1]

        # SHAP explanation
        shap_dict = explain_single(
            MODELS['explainer'], MODELS['scaler'], X_row
        )
        factors = top_factors(shap_dict, n=5)
        top_f = [
            {
                "feature": f,
                "shap_value": round(v, 4),
                "direction": "increases_approval" if v > 0 else "decreases_approval"
            }
            for f, v in factors
        ]

        group = "Privileged (Male)" if req.sensitive_attribute == 1 else "Unprivileged (Female)"
        note  = None
        if proba > 0.4 and proba < 0.6:
            note = "Borderline decision — fairness monitoring active"

        return AuditDecision(
            decision="APPROVED" if pred == 1 else "DENIED",
            confidence=round(float(proba), 4),
            group=group,
            top_factors=top_f,
            fairness_note=note
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/fairness-report")
def fairness_report():
    """Returns live fairness metrics from test set."""
    from src.data_loader import load_as_dataframe
    _, X_test, _, y_test, _, s_test = load_as_dataframe()

    baseline_preds  = MODELS['baseline'].predict(X_test)
    mitigated_preds = MODELS['mitigated'].predict(X_test)

    return {
        "baseline":  compute_fairness_metrics(y_test.values, baseline_preds, s_test.values),
        "mitigated": compute_fairness_metrics(y_test.values, mitigated_preds, s_test.values)
    }