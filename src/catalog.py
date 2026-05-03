from __future__ import annotations

from typing import Any, Callable

from algorithms.sklearn_wrappers import (
    create_baseline_knn,
    create_baseline_mlp,
    create_baseline_rf,
    create_baseline_svc,
    create_cascade_svc,
    create_cuml_knn,
    create_cuml_multi_gpu_knn,
    create_dualpipe_classifier,
    create_fsdp_mlp,
    create_hybrid_rf,
    create_thunder_svc,
)
from data.loaders import (
    load_breast_cancer_dataset,
    load_csv_classification_dataset,
    load_fraud_dataset,
)
from evaluation.configs import AlgorithmConfig, DatasetConfig, ScenarioConfig, SearchConfig


AlgorithmBuilder = Callable[[dict[str, Any]], Any]
DatasetLoader = Callable[[dict[str, Any]], Any]


ALGORITHM_REGISTRY: dict[str, dict[str, Any]] = {
    "BaselineMLP": {
        "builder": create_baseline_mlp,
        "default_parameters": {
            "hidden_layer_sizes": [100],
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "random_state": 42,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "hidden_layer_sizes": [(128,), (256, 128)],
                "learning_rate_init": [1e-3, 5e-4],
                "alpha": [1e-4, 1e-3],
            },
        },
    },
    "DualPipeClassifier": {
        "builder": create_dualpipe_classifier,
        "default_parameters": {
            "hidden_layer_sizes": [100],
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "random_state": 42,
            "num_chunks": 8,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "hidden_layer_sizes": [(128,), (256, 128)],
                "learning_rate_init": [1e-3, 5e-4],
                "num_chunks": [4, 8],
            },
        },
    },
    "FSDPMLP": {
        "builder": create_fsdp_mlp,
        "default_parameters": {
            "hidden_layer_sizes": [100],
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "random_state": 42,
            "use_fsdp": True,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "hidden_layer_sizes": [(128,), (256, 128)],
                "learning_rate_init": [1e-3, 5e-4],
                "alpha": [1e-4, 1e-3],
            },
        },
    },
    "BaselineSVC": {
        "builder": create_baseline_svc,
        "default_parameters": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "verbose": False,
            "random_state": 42,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "C": [0.5, 1.0, 2.0],
                "gamma": ["scale", "auto"],
            },
        },
    },
    "CascadeSVC": {
        "builder": create_cascade_svc,
        "default_parameters": {
            "fold_size": 100,
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "verbose": False,
            "random_state": 42,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "C": [0.5, 1.0, 2.0],
                "gamma": ["scale", "auto"],
                "fold_size": [100, 500],
            },
        },
    },
    "ThunderSVC": {
        "builder": create_thunder_svc,
        "default_parameters": {
            "kernel": "rbf",
            "C": 1.0,
            "gamma": "scale",
            "probability": True,
            "verbose": False,
            "random_state": 42,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "C": [0.5, 1.0, 2.0],
                "gamma": ["scale", "auto"],
            },
        },
    },
    "BaselineRF": {
        "builder": create_baseline_rf,
        "default_parameters": {
            "n_estimators": 100,
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5],
            },
        },
    },
    "HybridRF": {
        "builder": create_hybrid_rf,
        "default_parameters": {
            "n_estimators": 100,
            "max_depth": 5,
            "random_state": 42,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "n_estimators": [100, 200],
                "max_depth": [5, 10, 20],
            },
        },
    },
    "BaselineKNN": {
        "builder": create_baseline_knn,
        "default_parameters": {
            "n_neighbors": 5,
            "weights": "uniform",
            "algorithm": "auto",
            "leaf_size": 30,
            "p": 2,
            "metric": "minkowski",
            "metric_params": None,
            "n_jobs": 1,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "n_neighbors": [5, 11, 21],
                "weights": ["uniform", "distance"],
                "metric": ["minkowski", "manhattan"],
            },
        },
    },
    "CuMLKNN": {
        "builder": create_cuml_knn,
        "default_parameters": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "n_neighbors": [5, 11, 21],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
    },
    "CuMLMultiGPUKNN": {
        "builder": create_cuml_multi_gpu_knn,
        "default_parameters": {
            "n_neighbors": 5,
            "weights": "uniform",
            "metric": "euclidean",
            "n_workers": 2,
            "chunk_size": 100000,
        },
        "default_search": {
            "enabled": True,
            "strategy": "grid",
            "cv_folds": 3,
            "scoring": "f1_weighted",
            "param_grid": {
                "n_neighbors": [5, 11, 21],
                "weights": ["uniform", "distance"],
                "metric": ["euclidean", "manhattan"],
            },
        },
    },
}


DATASET_REGISTRY: dict[str, dict[str, Any]] = {
    "breast_cancer": {
        "loader": load_breast_cancer_dataset,
        "default_parameters": {"test_size": 0.25, "random_state": 42},
    },
    "fraud_csv": {
        "loader": load_fraud_dataset,
        "default_parameters": {
            "path": "datasets/creditcard.csv",
            "fraction": 1.0,
            "test_size": 0.2,
            "random_state": 42,
        },
    },
    "csv_classification": {
        "loader": load_csv_classification_dataset,
        "default_parameters": {
            "path": "",
            "target_column": "target",
            "fraction": 1.0,
            "test_size": 0.2,
            "random_state": 42,
        },
    },
    "breast_cancer_csv": {
        "loader": load_csv_classification_dataset,
        "default_parameters": {
            "path": "datasets/breast-cancer.csv",
            "target_column": "diagnosis",
            "drop_columns": ["id"],
            "test_size": 0.25,
            "random_state": 42,
        },
    },
    "covtype_csv": {
        "loader": load_csv_classification_dataset,
        "default_parameters": {
            "path": "datasets/covtype.csv",
            "header": "none",
            "target_column": -1,
            "test_size": 0.2,
            "random_state": 42,
        },
    },
    "airline_csv": {
        "loader": load_csv_classification_dataset,
        "default_parameters": {
            "path": "datasets/airline.csv",
            "target_column": "Cancelled",
            "test_size": 0.2,
            "random_state": 42,
        },
    },
    "higgs_csv": {
        "loader": load_csv_classification_dataset,
        "default_parameters": {
            "path": "datasets/HIGGS.csv",
            "header": "none",
            "target_column": 0,
            "test_size": 0.2,
            "random_state": 42,
        },
    },
}


def _require_registry_entry(name: str, registry: dict[str, dict[str, Any]], kind: str) -> dict[str, Any]:
    if name not in registry:
        available = ", ".join(sorted(registry))
        raise KeyError(f"Unknown {kind} '{name}'. Available {kind}s: {available}")
    return registry[name]


def build_algorithm_configs(entries: list[dict[str, Any]]) -> list[AlgorithmConfig]:
    if not entries:
        raise ValueError("At least one algorithm must be configured.")

    configs: list[AlgorithmConfig] = []
    for entry in entries:
        name = entry["name"]
        registry_entry = _require_registry_entry(name, ALGORITHM_REGISTRY, "algorithm")
        parameters = {
            **registry_entry["default_parameters"],
            **entry.get("parameters", {}),
        }
        search_defaults = registry_entry.get("default_search", {})
        search_overrides = entry.get("search", {})
        configs.append(
            AlgorithmConfig(
                name=name,
                parameters=parameters,
                implementation=registry_entry["builder"],
                search=SearchConfig(
                    enabled=bool(search_overrides.get("enabled", search_defaults.get("enabled", True))),
                    strategy=str(search_overrides.get("strategy", search_defaults.get("strategy", "grid"))),
                    param_grid={
                        **search_defaults.get("param_grid", {}),
                        **search_overrides.get("param_grid", {}),
                    },
                    cv_folds=int(search_overrides.get("cv_folds", search_defaults.get("cv_folds", 3))),
                    scoring=str(search_overrides.get("scoring", search_defaults.get("scoring", "f1_weighted"))),
                    n_iter=search_overrides.get("n_iter", search_defaults.get("n_iter")),
                    n_jobs=int(search_overrides.get("n_jobs", search_defaults.get("n_jobs", 1))),
                    random_state=int(search_overrides.get("random_state", search_defaults.get("random_state", 42))),
                ),
            )
        )
    return configs


def build_dataset_configs(entries: list[dict[str, Any]]) -> list[DatasetConfig]:
    if not entries:
        raise ValueError("At least one dataset must be configured.")

    configs: list[DatasetConfig] = []
    for entry in entries:
        loader_name = entry.get("loader", entry["name"])
        registry_entry = _require_registry_entry(loader_name, DATASET_REGISTRY, "dataset loader")
        parameters = {
            **registry_entry["default_parameters"],
            **entry.get("parameters", {}),
        }
        configs.append(
            DatasetConfig(
                name=entry["name"],
                loader_name=loader_name,
                load_data=registry_entry["loader"],
                base_params=parameters,
            )
        )
    return configs


def build_scenario_configs(entries: list[dict[str, Any]]) -> list[ScenarioConfig]:
    if not entries:
        return [ScenarioConfig(name="default", params={})]

    return [
        ScenarioConfig(
            name=entry["name"],
            params=entry.get("parameters", {}),
            description=entry.get("description", ""),
        )
        for entry in entries
    ]
