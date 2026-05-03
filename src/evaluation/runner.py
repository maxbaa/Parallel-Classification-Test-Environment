"""
runner.py
Central experiment controler that runs algorithms on datasets under different scenarios, collects results, and evaluates performance using defined metrics.
"""

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import json
import time
import traceback
import pandas as pd

from .configs import AlgorithmConfig, DatasetConfig, ScenarioConfig
from .monitoring import ResourceMonitor
from .metrics import compute_quality_metrics
from .tuning import fit_with_hyperparameter_search

@dataclass
class SingleRunResult:
    """Holds the result of a single experiment run."""
    algorithm: str
    dataset: str
    dataset_loader: str
    scenario: str
    run_index: int
    status: str
    error_type: str
    error_message: str
    n_train_samples: int
    n_test_samples: int
    search_strategy: str
    cv_folds: int
    evaluated_candidates: int
    best_cv_score: float
    best_params_json: str

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
                 repetitions: int=5, enable_gpu_monitoring: bool=True, continue_on_error: bool=True):
        self.algorithms = algorithms
        self.datasets = datasets
        self.scenarios = scenarios
        self.repetitions = repetitions
        self.enable_gpu_monitoring = enable_gpu_monitoring
        self.continue_on_error = continue_on_error
        self.failures: List[Dict[str, Any]] = []

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
                        run_number = run_idx + 1
                        print(f"Repetition {run_number}/{self.repetitions}")

                        try:
                            X_train, X_test, y_train, y_test = dataset_cfg.load_data(final_dataset_params)
                            n_train_samples = len(y_train)
                            n_test_samples = len(y_test)

                            with ResourceMonitor(enable_gpu=self.enable_gpu_monitoring) as monitor:
                                t0 = time.perf_counter()
                                base_estimator = algo_cfg.implementation(final_algo_params)
                                tuning_result = fit_with_hyperparameter_search(
                                    algorithm_config=algo_cfg,
                                    estimator=base_estimator,
                                    X_train=X_train,
                                    y_train=y_train,
                                )
                                t1 = time.perf_counter()

                            snapshot = monitor.get_snapshot()
                            train_time = t1 - t0
                            sec_per_sample = train_time / n_train_samples

                            model = tuning_result.estimator
                            qm = compute_quality_metrics(model, X_test, y_test)

                            run_result = SingleRunResult(
                                algorithm=algo_cfg.name,
                                dataset=dataset_cfg.name,
                                dataset_loader=dataset_cfg.loader_name,
                                scenario=scen_cfg.name,
                                run_index=run_number,
                                status="success",
                                error_type="",
                                error_message="",
                                n_train_samples=n_train_samples,
                                n_test_samples=n_test_samples,
                                search_strategy=tuning_result.search_strategy,
                                cv_folds=tuning_result.cv_folds,
                                evaluated_candidates=tuning_result.evaluated_candidates,
                                best_cv_score=tuning_result.best_score,
                                best_params_json=json.dumps(tuning_result.best_params, sort_keys=True),
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
                                gpu_max_gb=snapshot.gpu_max_gb,
                            )
                            results.append(asdict(run_result))
                        except Exception as exc:
                            failure = {
                                "algorithm": algo_cfg.name,
                                "dataset": dataset_cfg.name,
                                "dataset_loader": dataset_cfg.loader_name,
                                "scenario": scen_cfg.name,
                                "run_index": run_number,
                                "error_type": exc.__class__.__name__,
                                "error_message": str(exc),
                                "traceback": traceback.format_exc(),
                            }
                            self.failures.append(failure)

                            results.append(
                                asdict(
                                    SingleRunResult(
                                        algorithm=algo_cfg.name,
                                        dataset=dataset_cfg.name,
                                        dataset_loader=dataset_cfg.loader_name,
                                        scenario=scen_cfg.name,
                                        run_index=run_number,
                                        status="failed",
                                        error_type=failure["error_type"],
                                        error_message=failure["error_message"],
                                        n_train_samples=0,
                                        n_test_samples=0,
                                        search_strategy=algo_cfg.search.strategy if algo_cfg.search.enabled else "disabled",
                                        cv_folds=algo_cfg.search.cv_folds if algo_cfg.search.enabled else 0,
                                        evaluated_candidates=0,
                                        best_cv_score=float("NaN"),
                                        best_params_json="{}",
                                        accuracy=float("NaN"),
                                        precision=float("NaN"),
                                        recall=float("NaN"),
                                        f1=float("NaN"),
                                        roc_auc=float("NaN"),
                                        train_time_sec=float("NaN"),
                                        time_per_sample_sec=float("NaN"),
                                        cpu_avg=float("NaN"),
                                        cpu_max=float("NaN"),
                                        ram_avg_gb=float("NaN"),
                                        ram_max_gb=float("NaN"),
                                        gpu_max_gb=float("NaN"),
                                    )
                                )
                            )

                            if not self.continue_on_error:
                                raise

        return pd.DataFrame(results)
