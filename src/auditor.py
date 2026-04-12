"""
Run: python -m src.auditor
"""
import numpy as np
import pandas as pd

from src.data_loader    import load_as_dataframe
from src.model          import train_baseline
from src.bias_detector  import compute_fairness_metrics, print_fairness_report
from src.bias_mitigator import (train_reweighed, find_fair_thresholds,
                                predict_with_thresholds)
from src.explainer      import get_shap_explainer, explain_single, top_factors

def run_audit():
    print("\n" + "="*56)
    print("  AI HIRING FAIRNESS AUDITOR")
    print("="*56)

    # 1. Load
    print("\nLoading dataset...")
    X_train, X_test, y_train, y_test, s_train, s_test = load_as_dataframe()

    # 2. Baseline
    print("\nTraining baseline screening model...")
    baseline = train_baseline(X_train, y_train)
    y_pred_base = baseline.predict(X_test)
    metrics_base = compute_fairness_metrics(y_test.values, y_pred_base, s_test.values)
    print_fairness_report(metrics_base, "Baseline (Biased Model)")

    # 3. Mitigation A — Reweighing
    print("Applying Reweighing mitigation...")
    reweighed = train_reweighed(X_train, y_train, s_train)
    y_pred_rew = reweighed.predict(X_test)
    metrics_rew = compute_fairness_metrics(y_test.values, y_pred_rew, s_test.values)
    print_fairness_report(metrics_rew, "After Reweighing")

    # 4. Mitigation B — Threshold adjustment
    print("Finding fair thresholds...")
    t_priv, t_unpriv = find_fair_thresholds(baseline, X_test, y_test, s_test)
    print(f"  Male threshold: {t_priv:.2f} | Female threshold: {t_unpriv:.2f}")
    y_pred_thr = predict_with_thresholds(baseline, X_test, s_test, t_priv, t_unpriv)
    metrics_thr = compute_fairness_metrics(y_test.values, y_pred_thr, s_test.values)
    print_fairness_report(metrics_thr, "After Threshold Adjustment")

    # 5. Summary table
    print(f"\n{'='*68}")
    print("  BEFORE vs AFTER — HIRING BIAS REDUCTION SUMMARY")
    print(f"{'='*68}")
    rows = [
        ('Overall Accuracy',        'overall_accuracy'),
        ('Female Shortlist Rate',    'shortlist_rate_female'),
        ('Male Shortlist Rate',      'shortlist_rate_male'),
        ('Demographic Parity Diff',  'demographic_parity_diff'),
        ('Disparate Impact Ratio',   'disparate_impact_ratio'),
        ('Equal Opportunity Diff',   'equal_opportunity_diff'),
    ]
    print(f"  {'Metric':<30} {'Baseline':>10} {'Reweighed':>10} {'Threshold':>10}")
    print(f"  {'─'*64}")
    for label, key in rows:
        print(f"  {label:<30} "
              f"{metrics_base[key]:>10.3f} "
              f"{metrics_rew[key]:>10.3f} "
              f"{metrics_thr[key]:>10.3f}")
    print(f"{'='*68}")

    # 6. SHAP — explain a rejected female candidate
    print("\nExplaining a rejected female candidate (SHAP)...")
    female_rejections = X_test[(s_test.values == 0) & (y_pred_base == 0)]
    if len(female_rejections) > 0:
        sample = female_rejections.iloc[[0]]
        explainer, scaler = get_shap_explainer(baseline, X_train)
        shap_dict = explain_single(explainer, scaler, sample)
        factors   = top_factors(shap_dict, n=5)

        baseline_decision  = "SHORTLISTED" if baseline.predict(sample)[0] == 1 else "REJECTED"
        mitigated_decision = "SHORTLISTED" if reweighed.predict(sample)[0] == 1 else "REJECTED"

        print(f"\n  Candidate: Female applicant")
        print(f"  Baseline model decision  : {baseline_decision}")
        print(f"  Debiased model decision  : {mitigated_decision}")
        print(f"\n  Top factors driving baseline decision:")
        for feat, val in factors:
            direction = "supports shortlisting" if val > 0 else "drives rejection"
            print(f"    {feat:<28} {direction}  (SHAP: {val:+.4f})")

        if baseline_decision == "REJECTED" and mitigated_decision == "SHORTLISTED":
            print("\n  >>> Bias corrected: same candidate shortlisted after mitigation")

    print("\nAudit complete.\n")

if __name__ == "__main__":
    run_audit()