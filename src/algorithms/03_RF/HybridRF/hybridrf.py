from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


def _load_parallel_random_forest():
    module_path = Path(__file__).resolve().parent / "parallel_random_forest.py"
    spec = spec_from_file_location("hybrid_parallel_rf_module", module_path)
    module = module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module.RandomForest


ParallelRandomForest = _load_parallel_random_forest()


class HybridRF(ClassifierMixin, BaseEstimator):
    """Scikit-learn style wrapper for the hybrid parallel random forest approach."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=5,
        n_processes=1,
        random_state=42,
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.n_processes = n_processes
        self.random_state = random_state

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        self.model_ = ParallelRandomForest(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            random_state=self.random_state,
            n_processes=self.n_processes,
        )
        self.model_.fit(X, y_encoded.astype(int))
        return self

    def predict(self, X):
        class_indices = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[class_indices]

    def predict_proba(self, X):
        X = np.asarray(X)
        vote_totals = self._collect_weighted_votes(X)
        row_sums = vote_totals.sum(axis=1, keepdims=True)
        zero_rows = row_sums.squeeze(axis=1) == 0
        if np.any(zero_rows):
            vote_totals[zero_rows] = 1.0
            row_sums = vote_totals.sum(axis=1, keepdims=True)
        return vote_totals / row_sums

    def decision_function(self, X):
        proba = self.predict_proba(X)
        if proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def _collect_weighted_votes(self, X):
        vote_totals = np.zeros((len(X), len(self.classes_)), dtype=float)
        weights = np.asarray(self.model_.weights, dtype=float)

        for estimator_index, (tree, feature_indices) in enumerate(self.model_.estimators):
            X_subset = X if feature_indices is None else X[:, feature_indices]
            predictions = tree.predict(X_subset).astype(int)
            weight = weights[estimator_index] if estimator_index < len(weights) else 1.0
            vote_totals[np.arange(len(X)), predictions] += weight

        return vote_totals
