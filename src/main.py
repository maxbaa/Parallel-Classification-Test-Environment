from evaluation.runner import ExperimentRunner
from evaluation.configs import AlgorithmConfig, DatasetConfig, ScenarioConfig
from data.loaders import load_breast_cancer_dataset
from algorithms.sklearn_wrappers import create_cascade_svc

# 1) Algorithm configuration
algo = AlgorithmConfig(
    name="CascadeSVC_test",
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
    algorithms=[algo],
    datasets=[dataset],
    scenarios=[scenario],
    repetitions=3   # nur eine Wiederholung
)

df = runner.run()
print(df)
