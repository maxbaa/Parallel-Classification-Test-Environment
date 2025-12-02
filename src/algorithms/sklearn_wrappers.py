# src/algorithms/sklearn_wrappers.py
from sklearn.ensemble import RandomForestClassifier

def create_rf(params):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 50),
        n_jobs=params.get("n_workers", 1),
        random_state=42
    )
