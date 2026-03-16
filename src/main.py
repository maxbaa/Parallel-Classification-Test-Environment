from evaluation.runner import ExperimentRunner
from evaluation.configs import AlgorithmConfig, DatasetConfig, ScenarioConfig
from data.loaders import load_breast_cancer_dataset
from algorithms.sklearn_wrappers import create_baseline_svc, create_cascade_svc
from algorithms.sklearn_wrappers import create_baseline_mlp


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
    algorithms=[baseline_svm, cascade_svm, baseline_nn],
    datasets=[dataset],
    scenarios=[scenario],
    repetitions=50   # nur eine Wiederholung
)

df = runner.run()
print(df)
