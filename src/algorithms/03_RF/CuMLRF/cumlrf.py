from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    _CUML_IMPORT_ERROR = None
except ImportError as exc:
    CuMLRandomForestClassifier = None
    _CUML_IMPORT_ERROR = exc


def _as_numpy(value):
    if hasattr(value, "compute"):
        value = value.compute()
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


class CuMLRF(ClassifierMixin, BaseEstimator):
    """Single-GPU cuML random forest wrapper with sklearn-compatible methods."""

    def __init__(
        self,
        n_estimators=100,
        max_depth=16,
        max_features="sqrt",
        n_bins=128,
        random_state=42,
        output_type="numpy",
    ):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_features = max_features
        self.n_bins = n_bins
        self.random_state = random_state
        self.output_type = output_type

    def _require_cuml(self):
        if _CUML_IMPORT_ERROR is not None:
            raise ImportError(
                "cuML is required for CuMLRF. Install RAPIDS on the GPU server "
                "before running this algorithm."
            ) from _CUML_IMPORT_ERROR

    def fit(self, X, y):
        self._require_cuml()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.model_ = CuMLRandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            max_features=self.max_features,
            n_bins=self.n_bins,
            random_state=self.random_state,
            output_type=self.output_type,
        )
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        predictions = self.model_.predict(np.asarray(X, dtype=np.float32))
        return _as_numpy(predictions)

    def predict_proba(self, X):
        if not hasattr(self.model_, "predict_proba"):
            raise NotImplementedError("CuMLRF does not provide predict_proba in this environment.")
        probabilities = self.model_.predict_proba(np.asarray(X, dtype=np.float32))
        return _as_numpy(probabilities)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba
