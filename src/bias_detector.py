import numpy as np

def compute_fairness_metrics(y_true, y_pred, sensitive_attr):
    results = {}
    priv   = sensitive_attr == 1
    unpriv = sensitive_attr == 0

    # Shortlisting rates
    priv_rate   = y_pred[priv].mean()
    unpriv_rate = y_pred[unpriv].mean()

    results['shortlist_rate_male']   = float(priv_rate)
    results['shortlist_rate_female'] = float(unpriv_rate)
    results['demographic_parity_diff']  = float(priv_rate - unpriv_rate)
    results['disparate_impact_ratio']   = float(
        unpriv_rate / priv_rate if priv_rate > 0 else 0)

    # Equal opportunity (TPR per group)
    priv_tpr   = y_pred[priv   & (y_true == 1)].mean() if (priv   & (y_true == 1)).any() else 0
    unpriv_tpr = y_pred[unpriv & (y_true == 1)].mean() if (unpriv & (y_true == 1)).any() else 0
    results['equal_opportunity_diff'] = float(priv_tpr - unpriv_tpr)
    results['tpr_male']   = float(priv_tpr)
    results['tpr_female'] = float(unpriv_tpr)

    # Accuracy
    results['overall_accuracy'] = float((y_pred == y_true).mean())
    results['accuracy_male']    = float((y_pred[priv]   == y_true[priv]).mean())
    results['accuracy_female']  = float((y_pred[unpriv] == y_true[unpriv]).mean())

    return results


def print_fairness_report(metrics, label="Model"):
    print(f"\n{'='*56}")
    print(f"  Hiring Fairness Report — {label}")
    print(f"{'='*56}")
    print(f"  Overall Accuracy            : {metrics['overall_accuracy']:.3f}")
    print(f"  Male   Accuracy             : {metrics['accuracy_male']:.3f}")
    print(f"  Female Accuracy             : {metrics['accuracy_female']:.3f}")
    print(f"  {'─'*52}")
    print(f"  Shortlist Rate — Male       : {metrics['shortlist_rate_male']:.3f}")
    print(f"  Shortlist Rate — Female     : {metrics['shortlist_rate_female']:.3f}")
    print(f"  Demographic Parity Diff     : {metrics['demographic_parity_diff']:+.3f}  (ideal: 0.0)")
    print(f"  Disparate Impact Ratio      : {metrics['disparate_impact_ratio']:.3f}  (ideal: ≥0.8)")
    print(f"  {'─'*52}")
    print(f"  Equal Opportunity Diff      : {metrics['equal_opportunity_diff']:+.3f}  (ideal: 0.0)")
    print(f"  TPR Male                    : {metrics['tpr_male']:.3f}")
    print(f"  TPR Female                  : {metrics['tpr_female']:.3f}")

    # Plain-English verdict
    di = metrics['disparate_impact_ratio']
    dp = abs(metrics['demographic_parity_diff'])
    print(f"\n  Verdict: ", end="")
    if di < 0.8 or dp > 0.1:
        print(f"BIASED — female candidates {dp:.1%} less likely to be shortlisted")
    else:
        print("FAIR — shortlisting gap within acceptable bounds")
    print(f"{'='*56}\n")