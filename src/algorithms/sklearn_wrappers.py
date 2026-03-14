# src/algorithms/sklearn_wrappers.py
from algorithms.CascadeSVC.cascadesvc import CascadeSVC
from sklearn.ensemble import RandomForestClassifier

def create_rf(params):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 50),
        n_jobs=params.get("n_workers", 1),
        random_state=42
    )


def create_cascade_svc(params):
    return CascadeSVC(
        fold_size=params.get("fold_size", 10_000),
        verbose=params.get("verbose", False),
        C=params.get("C", 1.0),
        kernel=params.get("kernel", "rbf"),
        degree=params.get("degree", 3),
        gamma=params.get("gamma", "scale"),
        coef0=params.get("coef0", 0.0),
        probability=params.get("probability", True),
        random_state=params.get("random_state", 42),
    )
