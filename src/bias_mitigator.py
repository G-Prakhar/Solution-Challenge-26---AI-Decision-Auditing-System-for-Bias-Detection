import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier


def compute_sample_weights(y_train, sensitive_train):
    """Reweighing — upweights underrepresented (group × label) combinations."""
    n = len(y_train)
    weights = np.ones(n)
    for group in [0, 1]:
        for label in [0, 1]:
            mask = (sensitive_train == group) & (y_train == label)
            expected = (sensitive_train == group).mean() * (y_train == label).mean()
            actual   = mask.mean()
            if actual > 0:
                weights[mask] = expected / actual
    return weights


def train_reweighed(X_train, y_train, sensitive_train):
    weights = compute_sample_weights(y_train.values, sensitive_train.values)
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.1, random_state=42
        ))
    ])
    model.fit(X_train, y_train, clf__sample_weight=weights)
    return model


def find_fair_thresholds(model, X_val, y_val, sensitive_val):
    """Grid-search group-specific thresholds to equalise shortlisting rates."""
    proba = model.predict_proba(X_val)[:, 1]
    best_diff, best_t = float('inf'), (0.5, 0.5)

    for t_priv in np.arange(0.3, 0.75, 0.05):
        for t_unpriv in np.arange(0.3, 0.75, 0.05):
            preds = np.zeros(len(proba))
            preds[sensitive_val == 1] = (proba[sensitive_val == 1] >= t_priv).astype(int)
            preds[sensitive_val == 0] = (proba[sensitive_val == 0] >= t_unpriv).astype(int)
            diff = abs(preds[sensitive_val == 1].mean() - preds[sensitive_val == 0].mean())
            if diff < best_diff:
                best_diff, best_t = diff, (t_priv, t_unpriv)

    return best_t


def predict_with_thresholds(model, X, sensitive, t_priv=0.5, t_unpriv=0.5):
    proba = model.predict_proba(X)[:, 1]
    preds = np.zeros(len(proba), dtype=int)
    preds[sensitive.values == 1] = (proba[sensitive.values == 1] >= t_priv).astype(int)
    preds[sensitive.values == 0] = (proba[sensitive.values == 0] >= t_unpriv).astype(int)
    return preds