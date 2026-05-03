from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

try:
    from cuml.neighbors import KNeighborsClassifier as CuMLKNeighborsClassifier
    from cuml.ensemble import RandomForestClassifier as CuMLRandomForestClassifier
    _CUML_IMPORT_ERROR = None
except ImportError as exc:
    CuMLKNeighborsClassifier = None
    CuMLRandomForestClassifier = None
    _CUML_IMPORT_ERROR = exc

try:
    import dask.array as da
    from cuml.dask.neighbors import KNeighborsClassifier as DaskCuMLKNeighborsClassifier
    from dask_cuda import LocalCUDACluster
    from distributed import Client
    _CUML_DASK_IMPORT_ERROR = None
except ImportError as exc:
    da = None
    DaskCuMLKNeighborsClassifier = None
    LocalCUDACluster = None
    Client = None
    _CUML_DASK_IMPORT_ERROR = exc


def _as_numpy(value):
    if hasattr(value, "compute"):
        value = value.compute()
    if hasattr(value, "get"):
        value = value.get()
    return np.asarray(value)


class CuMLKNN(ClassifierMixin, BaseEstimator):
    """Single-GPU cuML KNN wrapper with a sklearn-compatible API."""

    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        metric="euclidean",
        algorithm="brute",
        output_type="numpy",
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.algorithm = algorithm
        self.output_type = output_type

    def _require_cuml(self):
        if _CUML_IMPORT_ERROR is not None:
            raise ImportError(
                "cuML is required for CuMLKNN. Install RAPIDS on the GPU server "
                "before running this algorithm."
            ) from _CUML_IMPORT_ERROR

    def fit(self, X, y):
        self._require_cuml()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.model_ = CuMLKNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            algorithm=self.algorithm,
            output_type=self.output_type,
        )
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        predictions = self.model_.predict(np.asarray(X, dtype=np.float32))
        return _as_numpy(predictions)

    def predict_proba(self, X):
        if not hasattr(self.model_, "predict_proba"):
            raise NotImplementedError("CuMLKNN does not provide predict_proba in this environment.")
        probabilities = self.model_.predict_proba(np.asarray(X, dtype=np.float32))
        return _as_numpy(probabilities)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba


class CuMLMultiGPUKNN(ClassifierMixin, BaseEstimator):
    """Multi-GPU cuML+dask KNN wrapper for one-process-per-GPU execution."""

    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        metric="euclidean",
        n_workers=2,
        chunk_size=100000,
        streams_per_handle=0,
        verbose=False,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.metric = metric
        self.n_workers = n_workers
        self.chunk_size = chunk_size
        self.streams_per_handle = streams_per_handle
        self.verbose = verbose

    def _require_cuml_dask(self):
        if _CUML_DASK_IMPORT_ERROR is not None:
            raise ImportError(
                "cuML Dask support is required for CuMLMultiGPUKNN. Install RAPIDS "
                "with dask-cuda/distributed on the GPU server before running this algorithm."
            ) from _CUML_DASK_IMPORT_ERROR

    def _cleanup_cluster(self):
        if hasattr(self, "client_") and self.client_ is not None:
            self.client_.close()
            self.client_ = None
        if hasattr(self, "cluster_") and self.cluster_ is not None:
            self.cluster_.close()
            self.cluster_ = None

    def __del__(self):
        self._cleanup_cluster()

    def _build_dask_arrays(self, X, y=None):
        row_chunk = max(1, min(int(self.chunk_size), len(X)))
        X_dask = da.from_array(np.asarray(X, dtype=np.float32), chunks=(row_chunk, X.shape[1]))
        if y is None:
            return X_dask, None
        y_dask = da.from_array(np.asarray(y), chunks=(row_chunk,))
        return X_dask, y_dask

    def fit(self, X, y):
        self._require_cuml_dask()
        self._cleanup_cluster()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        self.cluster_ = LocalCUDACluster(n_workers=self.n_workers, threads_per_worker=1)
        self.client_ = Client(self.cluster_)

        X_dask, y_dask = self._build_dask_arrays(X, y)
        self.model_ = DaskCuMLKNeighborsClassifier(
            client=self.client_,
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            metric=self.metric,
            batch_size=self.chunk_size,
            streams_per_handle=self.streams_per_handle,
            verbose=self.verbose,
        )
        self.model_.fit(X_dask, y_dask)
        self.classes_ = np.unique(y)
        return self

    def predict(self, X):
        X_dask, _ = self._build_dask_arrays(np.asarray(X, dtype=np.float32))
        predictions = self.model_.predict(X_dask)
        return _as_numpy(predictions)

    def predict_proba(self, X):
        if not hasattr(self.model_, "predict_proba"):
            raise NotImplementedError("CuMLMultiGPUKNN does not provide predict_proba in this environment.")
        X_dask, _ = self._build_dask_arrays(np.asarray(X, dtype=np.float32))
        probabilities = self.model_.predict_proba(X_dask)
        return _as_numpy(probabilities)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba


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
