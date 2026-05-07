"""
configs.py

This module contains basic Datamodels for:
- Algorithm configuration --> How do i build the model?
- Dataset configuration --> How do i load the data?
- Scenario configuration --> Which variation of the experiments do i want to run?
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Any, Tuple
import numpy as np

ArrayLike = np.ndarray


@dataclass
class AlgorithmConfig:
    """Configuration for an algorithm.

    Attributes:
        name (str): Name of the algorithm.
        parameters (Dict[str, Any]): Parameters for the algorithm --> hyperparameters (e.g., learning rate, number of trees).
        implementation (Callable): The implementation function of the algorithm.
    """
    name: str
    parameters: Dict[str, Any]
    implementation: Callable
    search: "SearchConfig" = field(default_factory=lambda: SearchConfig())
    allowed_scenarios: list[str] = field(default_factory=list)
    allowed_datasets: list[str] = field(default_factory=list)

@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        name (str): Name of the dataset.
        load_data (Callable): Function to load the dataset --> should return train and test splits (e.g., X_train, X_test, y_train, y_test).
        base_params (Dict[str, Any]): Base parameters for dataset loading (e.g. preprocessing options).
    """
    name: str
    loader_name: str
    load_data: Callable[[Dict[str, Any]], Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]]
    base_params: Dict[str, Any]
    source_name: str = ""
    split_label: str = ""

@dataclass
class ScenarioConfig:
    """Configuration for a scenario --> variation of the experiments.

    Attributes:
        name (str): Name of the scenario
        params (Dict[str, Any]): Parameters specific to the scenario.
    """
    name: str
    params: Dict[str, Any]
    description: str = ""


@dataclass
class ExperimentDefinition:
    """Top-level experiment definition used by the CLI entrypoint."""

    run_name: str
    repetitions: int
    output_dir: str
    enable_gpu_monitoring: bool
    continue_on_error: bool
    algorithms: list[AlgorithmConfig]
    datasets: list[DatasetConfig]
    scenarios: list[ScenarioConfig]


@dataclass
class SearchConfig:
    """Hyperparameter search configuration for one algorithm."""

    enabled: bool = True
    strategy: str = "grid"
    param_grid: Dict[str, list[Any]] = field(default_factory=dict)
    cv_folds: int = 3
    scoring: str = "f1_weighted"
    n_iter: int | None = None
    n_jobs: int = 1
    random_state: int = 42
