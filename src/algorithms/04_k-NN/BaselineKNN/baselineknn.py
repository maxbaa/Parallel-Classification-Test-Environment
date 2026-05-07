import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier


class BaselineKNN(ClassifierMixin, BaseEstimator):
    """Thin wrapper around sklearn's KNeighborsClassifier for baseline comparisons."""

    def __init__(
        self,
        n_neighbors=5,
        weights="uniform",
        algorithm="brute",
        leaf_size=30,
        metric="euclidean",
        metric_params=None,
        n_jobs=1,
    ):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        self.model_ = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights=self.weights,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            metric=self.metric,
            metric_params=self.metric_params,
            n_jobs=self.n_jobs,
        )
        self.model_.fit(X, y)
        self.classes_ = self.model_.classes_
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def decision_function(self, X):
        proba = self.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba

    def decision_function(self, X):
        return self.model_.predict_proba(X)[:, 1]
