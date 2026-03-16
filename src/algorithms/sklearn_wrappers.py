from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier


def _load_baseline_mlp():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "01_NN"
        / "BaselineMLP"
        / "baselinemlp.py"
    )
    spec = spec_from_file_location("baseline_mlp_module", baseline_module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.BaselineMLP


def _load_baseline_svc():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "02_SVM"
        / "BaselineSVC"
        / "baselinesvc.py"
    )
    spec = spec_from_file_location("baseline_svc_module", baseline_module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.BaselineSVC


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


BaselineMLP = _load_baseline_mlp()
BaselineSVC = _load_baseline_svc()
CascadeSVC = _load_cascade_svc()

def create_rf(params):
    return RandomForestClassifier(
        n_estimators=params.get("n_estimators", 50),
        n_jobs=params.get("n_workers", 1),
        random_state=42
    )


def create_baseline_mlp(params):
    return BaselineMLP(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
        activation=params.get("activation", "relu"),
        solver=params.get("solver", "adam"),
        alpha=params.get("alpha", 1e-4),
        batch_size=params.get("batch_size", "auto"),
        learning_rate=params.get("learning_rate", "constant"),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        power_t=params.get("power_t", 0.5),
        max_iter=params.get("max_iter", 200),
        shuffle=params.get("shuffle", True),
        tol=params.get("tol", 1e-4),
        verbose=params.get("verbose", False),
        warm_start=params.get("warm_start", False),
        momentum=params.get("momentum", 0.9),
        nesterovs_momentum=params.get("nesterovs_momentum", True),
        early_stopping=params.get("early_stopping", False),
        validation_fraction=params.get("validation_fraction", 0.1),
        beta_1=params.get("beta_1", 0.9),
        beta_2=params.get("beta_2", 0.999),
        epsilon=params.get("epsilon", 1e-8),
        n_iter_no_change=params.get("n_iter_no_change", 10),
        max_fun=params.get("max_fun", 15000),
        random_state=params.get("random_state", 42),
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


def create_baseline_svc(params):
    return BaselineSVC(
        C=params.get("C", 1.0),
        kernel=params.get("kernel", "rbf"),
        degree=params.get("degree", 3),
        gamma=params.get("gamma", "scale"),
        coef0=params.get("coef0", 0.0),
        shrinking=params.get("shrinking", True),
        probability=params.get("probability", True),
        tol=params.get("tol", 1e-3),
        cache_size=params.get("cache_size", 200),
        class_weight=params.get("class_weight"),
        verbose=params.get("verbose", False),
        max_iter=params.get("max_iter", -1),
        decision_function_shape=params.get("decision_function_shape", "ovr"),
        break_ties=params.get("break_ties", False),
        random_state=params.get("random_state", 42),
    )
