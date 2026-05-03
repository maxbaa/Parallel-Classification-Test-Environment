from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys


def _load_module(module_name: str, module_path: Path):
    spec = spec_from_file_location(module_name, module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_baseline_knn():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "04_k-NN"
        / "BaselineKNN"
        / "baselineknn.py"
    )
    module = _load_module("baseline_knn_module", baseline_module_path)
    return module.BaselineKNN


def _load_cuml_knn():
    module_path = (
        Path(__file__).resolve().parent
        / "04_k-NN"
        / "CuMLKNN"
        / "cumlknn.py"
    )
    module = _load_module("cuml_knn_module", module_path)
    return module.CuMLKNN, module.CuMLMultiGPUKNN, module.CuMLRF


def _load_baseline_mlp():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "01_NN"
        / "BaselineMLP"
        / "baselinemlp.py"
    )
    module = _load_module("baseline_mlp_module", baseline_module_path)
    return module.BaselineMLP


def _load_gpipe_mlp():
    module_path = (
        Path(__file__).resolve().parent
        / "01_NN"
        / "GPipeMLP"
        / "gpipemlp.py"
    )
    module = _load_module("gpipe_mlp_module", module_path)
    return module.GPipeMLP


def _load_fsdp_mlp():
    module_path = (
        Path(__file__).resolve().parent
        / "01_NN"
        / "FSDPMLP"
        / "fsdpmlp.py"
    )
    module = _load_module("fsdp_mlp_module", module_path)
    return module.FSDPMLP


def _load_baseline_rf():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "03_RF"
        / "BaselineRF"
        / "baselinerf.py"
    )
    module = _load_module("baseline_rf_module", baseline_module_path)
    return module.BaselineRF


def _load_hybrid_rf():
    hybrid_module_path = (
        Path(__file__).resolve().parent
        / "03_RF"
        / "HybridRF"
        / "hybridrf.py"
    )
    module = _load_module("hybrid_rf_module", hybrid_module_path)
    return module.HybridRF


def _load_baseline_svc():
    baseline_module_path = (
        Path(__file__).resolve().parent
        / "02_SVM"
        / "BaselineSVC"
        / "baselinesvc.py"
    )
    module = _load_module("baseline_svc_module", baseline_module_path)
    return module.BaselineSVC


def _load_cascade_svc():
    cascade_module_path = (
        Path(__file__).resolve().parent
        / "02_SVM"
        / "CascadeSVC"
        / "cascadesvc"
        / "cascadesvc.py"
    )
    module = _load_module("cascade_svc_module", cascade_module_path)
    return module.CascadeSVC


def _load_thunder_svc():
    thunder_module_path = (
        Path(__file__).resolve().parent
        / "02_SVM"
        / "ThunderSVC"
        / "thundersvc.py"
    )
    module = _load_module("thunder_svc_module", thunder_module_path)
    return module.ThunderSVC


def create_baseline_knn(params):
    baseline_knn = _load_baseline_knn()
    return baseline_knn(
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


def create_cuml_knn(params):
    cuml_knn, _, _ = _load_cuml_knn()
    return cuml_knn(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "uniform"),
        metric=params.get("metric", "euclidean"),
    )


def create_cuml_multi_gpu_knn(params):
    _, cuml_multi_gpu_knn, _ = _load_cuml_knn()
    return cuml_multi_gpu_knn(
        n_neighbors=params.get("n_neighbors", 5),
        weights=params.get("weights", "uniform"),
        metric=params.get("metric", "euclidean"),
        n_workers=params.get("n_workers", 2),
        chunk_size=params.get("chunk_size", 100000),
    )


def create_cuml_rf(params):
    _, _, cuml_rf = _load_cuml_knn()
    return cuml_rf(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 16),
        max_features=params.get("max_features", "sqrt"),
        n_bins=params.get("n_bins", 128),
        random_state=params.get("random_state", 42),
    )


def create_baseline_mlp(params):
    baseline_mlp = _load_baseline_mlp()
    return baseline_mlp(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
        activation=params.get("activation", "relu"),
        solver=params.get("solver", "adam"),
        alpha=params.get("alpha", 1e-4),
        batch_size=params.get("batch_size", "auto"),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        max_iter=params.get("max_iter", 200),
        shuffle=params.get("shuffle", True),
        tol=params.get("tol", 1e-4),
        verbose=params.get("verbose", False),
        momentum=params.get("momentum", 0.9),
        early_stopping=params.get("early_stopping", False),
        validation_fraction=params.get("validation_fraction", 0.1),
        n_iter_no_change=params.get("n_iter_no_change", 10),
        random_state=params.get("random_state", 42),
        device=params.get("device"),
    )


def create_gpipe_mlp(params):
    gpipe_mlp = _load_gpipe_mlp()
    return gpipe_mlp(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
        activation=params.get("activation", "relu"),
        solver=params.get("solver", "adam"),
        alpha=params.get("alpha", 1e-4),
        batch_size=params.get("batch_size", "auto"),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        max_iter=params.get("max_iter", 200),
        shuffle=params.get("shuffle", True),
        tol=params.get("tol", 1e-4),
        verbose=params.get("verbose", False),
        momentum=params.get("momentum", 0.9),
        early_stopping=params.get("early_stopping", False),
        validation_fraction=params.get("validation_fraction", 0.1),
        n_iter_no_change=params.get("n_iter_no_change", 10),
        random_state=params.get("random_state", 42),
        n_workers=params.get("n_workers", 2),
        chunks=params.get("chunks", 8),
        checkpoint=params.get("checkpoint", "except_last"),
    )


def create_fsdp_mlp(params):
    fsdp_mlp = _load_fsdp_mlp()
    return fsdp_mlp(
        hidden_layer_sizes=params.get("hidden_layer_sizes", (100,)),
        activation=params.get("activation", "relu"),
        solver=params.get("solver", "adam"),
        alpha=params.get("alpha", 1e-4),
        batch_size=params.get("batch_size", "auto"),
        learning_rate_init=params.get("learning_rate_init", 1e-3),
        max_iter=params.get("max_iter", 200),
        shuffle=params.get("shuffle", True),
        tol=params.get("tol", 1e-4),
        verbose=params.get("verbose", False),
        momentum=params.get("momentum", 0.9),
        early_stopping=params.get("early_stopping", False),
        validation_fraction=params.get("validation_fraction", 0.1),
        n_iter_no_change=params.get("n_iter_no_change", 10),
        random_state=params.get("random_state", 42),
        device=params.get("device"),
        n_workers=params.get("n_workers", 1),
        use_fsdp=params.get("use_fsdp", True),
        sync_module_states=params.get("sync_module_states", True),
    )


def create_baseline_rf(params):
    baseline_rf = _load_baseline_rf()
    return baseline_rf(
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
    hybrid_rf = _load_hybrid_rf()
    return hybrid_rf(
        n_estimators=params.get("n_estimators", 100),
        max_depth=params.get("max_depth", 5),
        n_processes=params.get("n_workers", 1),
        random_state=params.get("random_state", 42),
    )


def create_rf(params):
    return create_baseline_rf(params)


def create_cascade_svc(params):
    cascade_svc = _load_cascade_svc()
    return cascade_svc(
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
    baseline_svc = _load_baseline_svc()
    return baseline_svc(
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


def create_thunder_svc(params):
    thunder_svc = _load_thunder_svc()
    return thunder_svc(
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
        random_state=params.get("random_state", 42),
    )
