from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold

from .configs import AlgorithmConfig, SearchConfig


@dataclass
class TuningResult:
    estimator: Any
    best_params: dict[str, Any]
    best_score: float
    evaluated_candidates: int
    cv_folds: int
    search_strategy: str


def _normalize_param_grid(param_grid: dict[str, Any]) -> dict[str, list[Any]]:
    normalized: dict[str, list[Any]] = {}
    for key, value in param_grid.items():
        if isinstance(value, list):
            normalized[key] = value
        else:
            normalized[key] = [value]
    return normalized


def _resolve_cv_folds(y: Any, requested_folds: int) -> int:
    y_array = np.asarray(y)
    _, counts = np.unique(y_array, return_counts=True)
    if counts.size == 0:
        return max(2, requested_folds)
    minimum_count = int(counts.min())
    if minimum_count < 2:
        return 0
    return max(2, min(int(requested_folds), minimum_count))


def fit_with_hyperparameter_search(
    algorithm_config: AlgorithmConfig,
    estimator: Any,
    X_train: Any,
    y_train: Any,
) -> TuningResult:
    search_config: SearchConfig = algorithm_config.search
    param_grid = _normalize_param_grid(search_config.param_grid)

    if not search_config.enabled or not param_grid:
        estimator.fit(X_train, y_train)
        return TuningResult(
            estimator=estimator,
            best_params=estimator.get_params(deep=False),
            best_score=float("nan"),
            evaluated_candidates=0,
            cv_folds=0,
            search_strategy="disabled",
        )

    cv_folds = _resolve_cv_folds(y_train, search_config.cv_folds)
    if cv_folds < 2:
        estimator.fit(X_train, y_train)
        return TuningResult(
            estimator=estimator,
            best_params=estimator.get_params(deep=False),
            best_score=float("nan"),
            evaluated_candidates=0,
            cv_folds=0,
            search_strategy="disabled_insufficient_class_counts",
        )
    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=search_config.random_state,
    )

    search_strategy = search_config.strategy.lower()
    if search_strategy == "random":
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=param_grid,
            n_iter=search_config.n_iter or min(10, math.prod(len(values) for values in param_grid.values())),
            scoring=search_config.scoring,
            cv=cv,
            refit=True,
            n_jobs=search_config.n_jobs,
            random_state=search_config.random_state,
            error_score="raise",
        )
    else:
        search_strategy = "grid"
        search = GridSearchCV(
            estimator=estimator,
            param_grid=param_grid,
            scoring=search_config.scoring,
            cv=cv,
            refit=True,
            n_jobs=search_config.n_jobs,
            error_score="raise",
        )

    search.fit(X_train, y_train)
    evaluated_candidates = len(search.cv_results_["params"]) if hasattr(search, "cv_results_") else 0
    return TuningResult(
        estimator=search.best_estimator_,
        best_params=search.best_params_,
        best_score=float(search.best_score_),
        evaluated_candidates=evaluated_candidates,
        cv_folds=cv_folds,
        search_strategy=search_strategy,
    )
