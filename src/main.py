from evaluation.runner import ExperimentRunner
from evaluation.configs import AlgorithmConfig, DatasetConfig, ScenarioConfig
from data.loaders import load_breast_cancer_dataset
from algorithms.sklearn_wrappers import create_rf

# 1) Algorithm configuration
algo = AlgorithmConfig(
    name="RF_test",
    parameters={"n_estimators": 50},
    implementation=create_rf
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
    repetitions=10   # nur eine Wiederholung
)

df = runner.run()
print(df)
