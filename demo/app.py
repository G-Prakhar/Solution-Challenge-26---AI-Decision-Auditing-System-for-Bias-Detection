import streamlit as st
import requests, json
import pandas as pd
import matplotlib.pyplot as plt

API_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Decision Auditor", page_icon="⚖️", layout="wide")
st.title("⚖️ AI Decision Auditing System")
st.caption("Unbiased, explainable ML decisions with real-time fairness monitoring")

tab1, tab2 = st.tabs(["🔍 Single Decision Audit", "📊 Fairness Dashboard"])

# ── Tab 1: Single prediction ───────────────────────────────────────────────────
with tab1:
    st.subheader("Audit a Loan Decision")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Applicant Profile**")
        duration    = st.slider("Loan Duration (months)", 6, 72, 24)
        amount      = st.number_input("Credit Amount", 500, 20000, 3000)
        age         = st.slider("Age", 18, 75, 35)
        employment  = st.selectbox("Employment", [1, 2, 3, 4, 5])
        sex         = st.radio("Sex (protected attribute)", [1, 0],
                               format_func=lambda x: "Male" if x == 1 else "Female")
        use_fair    = st.toggle("Use debiased model", value=True)

    with col2:
        if st.button("🔎 Audit This Decision", type="primary"):
            payload = {
                "features": {
                    "duration": duration,
                    "credit_amount": amount,
                    "age": age,
                    "employment": employment,
                    "sex": sex,
                },
                "sensitive_attribute": sex,
                "use_mitigated": use_fair
            }
            resp = requests.post(f"{API_URL}/predict", json=payload)
            if resp.ok:
                result = resp.json()
                decision = result['decision']
                conf     = result['confidence']

                color = "green" if decision == "APPROVED" else "red"
                st.markdown(f"### :{color}[{decision}]")
                st.metric("Confidence", f"{conf:.1%}")
                st.caption(f"Group: {result['group']}")
                if result.get('fairness_note'):
                    st.warning(result['fairness_note'])

                st.markdown("**Top decision factors (SHAP)**")
                factors = result['top_factors']
                feat_df = pd.DataFrame(factors)
                feat_df['color'] = feat_df['direction'].map({
                    'increases_approval': '#2e7d32',
                    'decreases_approval': '#c62828'
                })
                fig, ax = plt.subplots(figsize=(5, 3))
                bars = ax.barh(feat_df['feature'], feat_df['shap_value'],
                               color=feat_df['color'])
                ax.axvline(0, color='gray', linewidth=0.8)
                ax.set_xlabel("SHAP value")
                ax.set_title("Feature Contributions")
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.error(f"API error: {resp.text}")

# ── Tab 2: Fairness Dashboard ──────────────────────────────────────────────────
with tab2:
    st.subheader("Live Fairness Metrics")
    if st.button("Refresh Metrics"):
        resp = requests.get(f"{API_URL}/fairness-report")
        if resp.ok:
            data = resp.json()
            b = data['baseline']
            m = data['mitigated']

            col1, col2, col3 = st.columns(3)
            col1.metric("Demographic Parity Diff",
                        f"{m['demographic_parity_diff']:+.3f}",
                        delta=f"{m['demographic_parity_diff'] - b['demographic_parity_diff']:+.3f}",
                        delta_color="inverse")
            col2.metric("Disparate Impact Ratio",
                        f"{m['disparate_impact_ratio']:.3f}",
                        delta=f"{m['disparate_impact_ratio'] - b['disparate_impact_ratio']:+.3f}")
            col3.metric("Equal Opportunity Diff",
                        f"{m['equal_opportunity_diff']:+.3f}",
                        delta=f"{m['equal_opportunity_diff'] - b['equal_opportunity_diff']:+.3f}",
                        delta_color="inverse")

            # Comparison chart
            labels  = ['Demogr. Parity Diff', 'Disparate Impact', 'Equal Opp. Diff']
            before  = [abs(b['demographic_parity_diff']),
                       abs(1 - b['disparate_impact_ratio']),
                       abs(b['equal_opportunity_diff'])]
            after   = [abs(m['demographic_parity_diff']),
                       abs(1 - m['disparate_impact_ratio']),
                       abs(m['equal_opportunity_diff'])]

            fig, ax = plt.subplots(figsize=(7, 3))
            x = range(len(labels))
            ax.bar([i - 0.2 for i in x], before, width=0.4, label='Baseline', color='#c62828', alpha=0.8)
            ax.bar([i + 0.2 for i in x], after,  width=0.4, label='Mitigated', color='#2e7d32', alpha=0.8)
            ax.set_xticks(list(x))
            ax.set_xticklabels(labels)
            ax.set_ylabel("Bias magnitude (lower = fairer)")
            ax.set_title("Bias: Before vs After Mitigation")
            ax.legend()
            plt.tight_layout()
            st.pyplot(fig)