"""
runner.py
Central experiment controler that runs algorithms on datasets under different scenarios, collects results, and evaluates performance using defined metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import time
import pandas as pd

from .configs import AlgorithmConfig, DatasetConfig, ScenarioConfig
from .monitoring import ResourceMonitor
from .metrics import compute_quality_metrics

@dataclass
class SingleRunResult:
    """Holds the result of a single experiment run."""
    algorithm: str
    dataset: str
    scenario: str
    run_index: int

    # Quality metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float

    # Timing
    train_time_sec: float
    time_per_sample_sec: float

    # Resources
    cpu_avg: float
    cpu_max: float
    ram_avg_gb: float
    ram_max_gb: float
    gpu_max_gb: float

class ExperimentRunner:
    """
    Class to run experiments with specified algorithms, datasets, and scenarios.
    """

    def __init__(self, algorithms: List[AlgorithmConfig], datasets: List[DatasetConfig], scenarios: List[ScenarioConfig], 
                 repetitions: int=5, enable_gpu_monitoring: bool=True):
        self.algorithms = algorithms
        self.datasets = datasets
        self.scenarios = scenarios
        self.repetitions = repetitions
        self.enable_gpu_monitoring = enable_gpu_monitoring

    def run(self) -> pd.DataFrame:
        results: List[Dict[str, Any]] = []

        for dataset_cfg in self.datasets:
            for algo_cfg in self.algorithms:
                for scen_cfg in self.scenarios:

                    print(f"\nRunning {algo_cfg.name} on {dataset_cfg.name} "
                          f"under scenario '{scen_cfg.name}'")

                    final_dataset_params = {
                        **dataset_cfg.base_params,
                        **scen_cfg.params
                    }

                    final_algo_params = {
                        **algo_cfg.parameters,
                        **scen_cfg.params
                    }

                    for run_idx in range(self.repetitions):
                        print(f"Repetition {run_idx+1}/{self.repetitions}")

                        X_train, X_test, y_train, y_test = dataset_cfg.load_data(final_dataset_params)
                        n_train_samples = len(y_train)

                        model = algo_cfg.implementation(final_algo_params)

                        with ResourceMonitor(enable_gpu=self.enable_gpu_monitoring) as monitor:
                            t0 = time.perf_counter()
                            model.fit(X_train, y_train)
                            t1 = time.perf_counter()

                        snapshot = monitor.get_snapshot()
                        train_time = t1 - t0
                        sec_per_sample = train_time / n_train_samples

                        qm = compute_quality_metrics(model, X_test, y_test)

                        run_result = SingleRunResult(
                            algorithm=algo_cfg.name,
                            dataset=dataset_cfg.name,
                            scenario=scen_cfg.name,
                            run_index=run_idx,

                            accuracy=qm.accuracy,
                            precision=qm.precision,
                            recall=qm.recall,
                            f1=qm.f1,
                            roc_auc=qm.roc_auc if qm.roc_auc is not None else float("NaN"),

                            train_time_sec=train_time,
                            time_per_sample_sec=sec_per_sample,

                            cpu_avg=snapshot.cpu_avg,
                            cpu_max=snapshot.cpu_max,
                            ram_avg_gb=snapshot.ram_avg_gb,
                            ram_max_gb=snapshot.ram_max_gb,
                            gpu_max_gb=snapshot.gpu_max_gb
                        )

                        results.append(asdict(run_result))

        return pd.DataFrame(results)