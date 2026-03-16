from evaluation.runner import ExperimentRunner
from evaluation.configs import AlgorithmConfig, DatasetConfig, ScenarioConfig
from data.loaders import load_breast_cancer_dataset
from algorithms.sklearn_wrappers import create_baseline_svc, create_cascade_svc
from algorithms.sklearn_wrappers import create_baseline_mlp
from algorithms.sklearn_wrappers import create_baseline_rf
from algorithms.sklearn_wrappers import create_hybrid_rf
from algorithms.sklearn_wrappers import create_baseline_knn


# 1) Algorithm configurations

# Neural Network (MLP) configuration

baseline_nn = AlgorithmConfig(
    name="BaselineMLP",
    parameters={
        "hidden_layer_sizes": (100,),
        "activation": "relu",
        "solver": "adam",
        "max_iter": 200,
        "random_state": 42,
    },
    implementation=create_baseline_mlp,
)

# SVM configurations

baseline_svm = AlgorithmConfig(
    name="BaselineSVC",
    parameters={
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "verbose": False,
        "random_state": 42,
    },
    implementation=create_baseline_svc,
)

cascade_svm = AlgorithmConfig(
    name="CascadeSVC",
    parameters={
        "fold_size": 100,
        "kernel": "rbf",
        "C": 1.0,
        "gamma": "scale",
        "probability": True,
        "verbose": False,
        "random_state": 42,
    },
    implementation=create_cascade_svc,
)

# Random Forest configurations

baseline_rf = AlgorithmConfig(
    name="BaselineRF",
    parameters={
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "random_state": 42,
    },
    implementation=create_baseline_rf,
)

hybrid_rf = AlgorithmConfig(
    name="HybridRF",
    parameters={
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    },
    implementation=create_hybrid_rf,
)

# k-NN configurations

baseline_knn = AlgorithmConfig(
    name="BaselineKNN",
    parameters={
        "n_neighbors": 5,
        "weights": "uniform",
        "algorithm": "auto",
        "leaf_size": 30,
        "p": 2,
        "metric": "minkowski",
        "metric_params": None,
        "n_jobs": 1,
    },
    implementation=create_baseline_knn,
)

# 2) Dataset configuration
dataset = DatasetConfig(
    name="breast_cancer",
    load_data=load_breast_cancer_dataset,
    base_params={"test_size": 0.25}
)

# 3) Scenario configuration
scenario = ScenarioConfig(
    name="seq",
    params={"n_workers": 1}
)

# 4) Runner
runner = ExperimentRunner(
    algorithms=[baseline_svm, cascade_svm, baseline_nn, baseline_rf, hybrid_rf, baseline_knn],
    datasets=[dataset],
    scenarios=[scenario],
    repetitions=3   # nur eine Wiederholung
)

df = runner.run()
print(df)
