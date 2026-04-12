from pydantic import BaseModel
from typing import Dict, List, Optional

class PredictRequest(BaseModel):
    features: Dict[str, float]
    sensitive_attribute: int  # 1=privileged, 0=unprivileged
    use_mitigated: bool = True

class AuditDecision(BaseModel):
    decision: str            # "APPROVED" or "DENIED"
    confidence: float
    group: str
    top_factors: List[Dict]  # [{feature, shap_value, direction}]
    fairness_note: Optional[str]