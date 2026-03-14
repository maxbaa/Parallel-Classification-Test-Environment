from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


def _load_cascade_svc():
    cascade_module_path = (
        Path(__file__).resolve().parent
        / "02_SVM"
        / "CascadeSVC"
        / "cascadesvc"
        / "cascadesvc.py"
    )
    spec = spec_from_file_location("cascade_svc_module", cascade_module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.CascadeSVC


CascadeSVC = _load_cascade_svc()

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
