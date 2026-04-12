from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def train_baseline(X_train, y_train):
    """
    GradientBoosting — better than LogisticRegression for tabular hiring data,
    and more realistic (companies actually use tree-based models for screening).
    """
    model = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        ))
    ])
    model.fit(X_train, y_train)
    return model