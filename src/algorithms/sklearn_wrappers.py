from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

def _load_baseline_knn():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "04_k-NN"
        / "BaselineKNN"
        / "baselineknn.py"
    )
    spec = spec_from_file_location("baseline_knn_module", baseline_module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.BaselineKNN


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


def _load_baseline_rf():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "03_RF"
        / "BaselineRF"
        / "baselinerf.py"
    )
    spec = spec_from_file_location("baseline_rf_module", baseline_module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.BaselineRF


def _load_hybrid_rf():
    hybrid_module_path = (
        Path(__file__).resolve().parent
        / "03_RF"
        / "HybridRF"
        / "hybridrf.py"
    )
    spec = spec_from_file_location("hybrid_rf_module", hybrid_module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.HybridRF


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


BaselineKNN = _load_baseline_knn()
BaselineMLP = _load_baseline_mlp()
BaselineRF = _load_baseline_rf()
HybridRF = _load_hybrid_rf()
BaselineSVC = _load_baseline_svc()
CascadeSVC = _load_cascade_svc()

def create_baseline_knn(params):
    return BaselineKNN(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "uniform"),
        algorithm=params.get("algorithm", "auto"),
        leaf_size=params.get("leaf_size", 30),
        p=params.get("p", 2),
        metric=params.get("metric", "minkowski"),
        metric_params=params.get("metric_params"),
        n_jobs=params.get("n_workers", 1),
    )


def create_knn(params):
    return create_baseline_knn(params)


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


def create_baseline_rf(params):
    return BaselineRF(
        n_estimators=params.get("n_estimators", 100),
        criterion=params.get("criterion", "gini"),
        max_depth=params.get("max_depth"),
        min_samples_split=params.get("min_samples_split", 2),
        min_samples_leaf=params.get("min_samples_leaf", 1),
        min_weight_fraction_leaf=params.get("min_weight_fraction_leaf", 0.0),
        max_features=params.get("max_features", "sqrt"),
        max_leaf_nodes=params.get("max_leaf_nodes"),
        min_impurity_decrease=params.get("min_impurity_decrease", 0.0),
        bootstrap=params.get("bootstrap", True),
        oob_score=params.get("oob_score", False),
        n_jobs=params.get("n_workers", 1),
        random_state=params.get("random_state", 42),
        verbose=params.get("verbose", 0),
        warm_start=params.get("warm_start", False),
        class_weight=params.get("class_weight"),
        ccp_alpha=params.get("ccp_alpha", 0.0),
        max_samples=params.get("max_samples"),
    )


def create_hybrid_rf(params):
    return HybridRF(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 5),
        n_processes=params.get("n_workers", 1),
        random_state=params.get("random_state", 42),
    )


def create_rf(params):
    return create_baseline_rf(params)


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
