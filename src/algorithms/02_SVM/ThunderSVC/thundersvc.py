import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
import inspect

try:
    from thundersvm import SVC as ThunderSVMClassifier
    _THUNDERSVM_IMPORT_ERROR = None
except ImportError as exc:
    ThunderSVMClassifier = None
    _THUNDERSVM_IMPORT_ERROR = exc


class ThunderSVC(ClassifierMixin, BaseEstimator):
    """Thin wrapper around ThunderSVM's SVC for accelerated baseline comparisons."""

    def __init__(
        self,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=True,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        random_state=42,
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.random_state = random_state

    def _require_thundersvm(self):
        if _THUNDERSVM_IMPORT_ERROR is not None:
            raise ImportError(
                "ThunderSVM is required for ThunderSVC. Install project "
                "dependencies again so that 'thundersvm' is available."
            ) from _THUNDERSVM_IMPORT_ERROR

    def _build_model(self):
        self._require_thundersvm()
        thunder_kwargs = {
            "C": self.C,
            "kernel": self.kernel,
            "degree": self.degree,
            "gamma": self._resolved_gamma,
            "coef0": self.coef0,
            "shrinking": self.shrinking,
            "probability": self.probability,
            "tol": self.tol,
            "cache_size": self.cache_size,
            "class_weight": self.class_weight,
            "verbose": self.verbose,
            "max_iter": self.max_iter,
            "decision_function_shape": self.decision_function_shape,
            "random_state": self.random_state,
        }
        valid_parameters = inspect.signature(ThunderSVMClassifier.__init__).parameters
        unsupported = sorted(key for key in thunder_kwargs if key not in valid_parameters)
        if unsupported:
            raise TypeError(
                "ThunderSVM does not support the configured parameters: "
                + ", ".join(unsupported)
            )
        self.execution_mode_ = "thundersvm"
        return ThunderSVMClassifier(**thunder_kwargs)

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if self.gamma == "scale":
            self._resolved_gamma = 1.0 / (X.shape[1] * X.var())
        elif self.gamma == "auto":
            self._resolved_gamma = 1.0 / X.shape[1]
        else:
            self._resolved_gamma = self.gamma

        self.model_ = self._build_model()
        self.model_.fit(X, y)
        self.classes_ = np.unique(y)
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)
