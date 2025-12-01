"""
configs.py

This module contains basic Datamodels for:
- Algorithm configuration --> How do i build the model?
- Dataset configuration --> How do i load the data?
- Scenario configuration --> Which variation of the experiments do i want to run?
"""

from dataclasses import dataclass
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

@dataclass
class DatasetConfig:
    """Configuration for a dataset.

    Attributes:
        name (str): Name of the dataset.
        load_data (Callable): Function to load the dataset --> should return train and test splits (e.g., X_train, X_test, y_train, y_test).
        base_params (Dict[str, Any]): Base parameters for dataset loading (e.g. preprocessing options).
    """
    name: str
    load_data: Callable[[Dict[str, Any]], Tuple[ArrayLike, ArrayLike, ArrayLike, ArrayLike]]
    base_params: Dict[str, Any]

@dataclass
class ScenarioConfig:
    """Configuration for a scenario --> variation of the experiments.

    Attributes:
        name (str): Name of the scenario
        params (Dict[str, Any]): Parameters specific to the scenario.
    """
    name: str
    params: Dict[str, Any]