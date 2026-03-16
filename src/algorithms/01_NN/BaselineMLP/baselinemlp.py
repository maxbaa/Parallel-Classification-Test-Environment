from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neural_network import MLPClassifier


class BaselineMLP(ClassifierMixin, BaseEstimator):
    """Thin wrapper around sklearn's MLPClassifier for baseline comparisons."""

    def __init__(
        self,
        hidden_layer_sizes=(100,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=1e-3,
        power_t=0.5,
        max_iter=200,
        shuffle=True,
        tol=1e-4,
        verbose=False,
        warm_start=False,
        momentum=0.9,
        nesterovs_momentum=True,
        early_stopping=False,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,
        max_fun=15000,
        random_state=42,
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.learning_rate_init = learning_rate_init
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.warm_start = warm_start
        self.momentum = momentum
        self.nesterovs_momentum = nesterovs_momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.n_iter_no_change = n_iter_no_change
        self.max_fun = max_fun
        self.random_state = random_state

    def fit(self, X, y):
        self.model_ = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self.activation,
            solver=self.solver,
            alpha=self.alpha,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            learning_rate_init=self.learning_rate_init,
            power_t=self.power_t,
            max_iter=self.max_iter,
            shuffle=self.shuffle,
            tol=self.tol,
            verbose=self.verbose,
            warm_start=self.warm_start,
            momentum=self.momentum,
            nesterovs_momentum=self.nesterovs_momentum,
            early_stopping=self.early_stopping,
            validation_fraction=self.validation_fraction,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
            epsilon=self.epsilon,
            n_iter_no_change=self.n_iter_no_change,
            max_fun=self.max_fun,
            random_state=self.random_state,
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        if hasattr(self.model_, "decision_function"):
            return self.model_.decision_function(X)
        return self.model_.predict_proba(X)[:, 1]

