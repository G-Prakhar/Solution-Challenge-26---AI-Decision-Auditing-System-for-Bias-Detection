# AI Decision Auditing System — Unbiased AI

> Detect, explain, and mitigate bias in ML decision systems. Built for loan approval, credit scoring, and hiring pipelines.

---

## What it does

1. Trains a baseline ML model on a loan approval dataset
2. Detects bias using demographic parity, equal opportunity, and disparate impact metrics
3. Applies reweighing and threshold adjustment to reduce bias
4. Outputs explainable decisions using SHAP feature importance
5. Exposes everything via a FastAPI REST API with a Streamlit demo UI

---

## Project structure

```
ai-decision-audit/
├── data/
│   └── german_credit.csv
├── src/
│   ├── __init__.py
│   ├── data_loader.py        # Load + preprocess German Credit dataset
│   ├── model.py              # Train baseline logistic regression model
│   ├── bias_detector.py      # Compute fairness metrics
│   ├── bias_mitigator.py     # Reweighing + threshold adjustment
│   ├── explainer.py          # SHAP wrapper for decision explanations
│   └── auditor.py            # Orchestrates full audit pipeline
├── api/
│   ├── main.py               # FastAPI app
│   └── schemas.py            # Pydantic request/response models
├── demo/
│   └── app.py                # Streamlit frontend
├── notebooks/
│   └── exploration.ipynb
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the core audit (CLI)

```bash
python -m src.auditor
```

This will print a full before/after bias report directly in the terminal.

### 3. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

Interactive API docs available at `http://localhost:8000/docs`

### 4. Launch the demo UI

```bash
streamlit run demo/app.py
```

---

## Tech stack

| Layer | Tool |
|---|---|
| Dataset | UCI German Credit via `aif360` |
| ML model | `scikit-learn` LogisticRegression |
| Fairness detection | `aif360`, `fairlearn` |
| Explainability | `shap` |
| API | `FastAPI` + `uvicorn` |
| Demo UI | `Streamlit` |

---

## Fairness metrics

| Metric | What it measures | Ideal value |
|---|---|---|
| Demographic parity difference | Gap in approval rates between groups | 0.0 |
| Disparate impact ratio | Approval rate ratio (unprivileged / privileged) | ≥ 0.8 |
| Equal opportunity difference | Gap in true positive rates between groups | 0.0 |

---

## Bias mitigation techniques

**Reweighing (pre-processing)** — Assigns higher sample weights to underrepresented (group × label) combinations before training, so the model learns an equal representation of outcomes across groups.

**Threshold adjustment (post-processing)** — Finds group-specific decision thresholds that minimise the demographic parity gap without retraining the model. Useful when you cannot modify the training pipeline.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check |
| `POST` | `/predict` | Audit a single decision with SHAP explanation |
| `GET` | `/fairness-report` | Live fairness metrics (baseline vs mitigated) |

### Example request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "features": {
      "duration": 24,
      "credit_amount": 3000,
      "age": 35
    },
    "sensitive_attribute": 0,
    "use_mitigated": true
  }'
```

### Example response

```json
{
  "decision": "APPROVED",
  "confidence": 0.7231,
  "group": "Unprivileged (Female)",
  "top_factors": [
    { "feature": "credit_history", "shap_value": 0.312, "direction": "increases_approval" },
    { "feature": "duration",       "shap_value": -0.18, "direction": "decreases_approval" }
  ],
  "fairness_note": null
}
```

---

## Sample audit output

```
==================================================
  Fairness Report — Baseline (Before Mitigation)
==================================================
  Overall Accuracy          : 0.743
  Privileged   Accuracy     : 0.761
  Unprivileged Accuracy     : 0.689
  ─────────────────────────────────────
  Demographic Parity Diff   : +0.142  ← (ideal: 0.0)
  Disparate Impact Ratio    : 0.694   ← (ideal: ≥0.8)
  Equal Opportunity Diff    : +0.119  ← (ideal: 0.0)
==================================================

==================================================
  Fairness Report — After Reweighing
==================================================
  Overall Accuracy          : 0.731
  Privileged   Accuracy     : 0.742
  Unprivileged Accuracy     : 0.714
  ─────────────────────────────────────
  Demographic Parity Diff   : +0.031  ← (ideal: 0.0)
  Disparate Impact Ratio    : 0.923   ← (ideal: ≥0.8)
  Equal Opportunity Diff    : +0.027  ← (ideal: 0.0)
==================================================
```

---

## Dataset

The [UCI German Credit dataset](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)) contains 1,000 loan applicants with 20 features including credit history, loan duration, employment status, and age.

**Protected attribute:** `sex` (1 = male / privileged, 0 = female / unprivileged)  
**Label:** `credit_risk` (1 = good credit / approved, 0 = bad credit / denied)

The dataset is loaded directly via `aif360` — no manual download required.

---

## Extending this system

- **More datasets** — swap in COMPAS (recidivism) or Adult Income by replacing `data_loader.py`
- **More metrics** — add average odds difference or Theil index in `bias_detector.py`
- **More mitigation** — plug in `aif360`'s Adversarial Debiasing or Prejudice Remover
- **Production** — add JWT auth, per-request audit logging to Postgres, and batch `/audit` endpoint
- **Compliance mapping** — map metrics to EU AI Act Article 10 and the EEOC 80% rule

---

## Requirements

```
pandas
numpy
scikit-learn
aif360
fairlearn
shap
fastapi
uvicorn
streamlit
matplotlib
```

---

## License

Apache License 2.0