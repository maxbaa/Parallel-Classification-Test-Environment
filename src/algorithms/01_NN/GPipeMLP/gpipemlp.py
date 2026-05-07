import copy
import inspect

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torchgpipe import GPipe
    _GPipe_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None
    GPipe = None
    _GPipe_IMPORT_ERROR = exc


class GPipeMLP(ClassifierMixin, BaseEstimator):
    """PyTorch MLP wrapped with torchgpipe.GPipe for multi-GPU pipeline parallelism."""

    def __init__(
        self,
        hidden_layer_sizes=(512, 512, 256),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=2048,
        learning_rate_init=1e-3,
        max_iter=200,
        shuffle=True,
        tol=1e-4,
        verbose=False,
        momentum=0.9,
        early_stopping=False,
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        n_workers=2,
        chunks=8,
        checkpoint="except_last",
    ):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate_init = learning_rate_init
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.tol = tol
        self.verbose = verbose
        self.momentum = momentum
        self.early_stopping = early_stopping
        self.validation_fraction = validation_fraction
        self.n_iter_no_change = n_iter_no_change
        self.random_state = random_state
        self.n_workers = n_workers
        self.chunks = chunks
        self.checkpoint = checkpoint

    def _require_gpipe(self):
        if _GPipe_IMPORT_ERROR is not None:
            raise ImportError(
                "torchgpipe is required for GPipeMLP. Install project dependencies "
                "again so that 'torchgpipe' is available."
            ) from _GPipe_IMPORT_ERROR

    def _get_activation(self):
        activations = {
            "identity": None,
            "logistic": nn.Sigmoid,
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
        }
        if self.activation not in activations:
            raise ValueError(f"Unsupported activation '{self.activation}'.")
        return activations[self.activation]

    def _get_batch_size(self, n_samples):
        if self.batch_size == "auto":
            return min(200, n_samples)
        return max(1, int(self.batch_size))

    def _encode_targets(self, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._is_binary = len(self.classes_) == 2
        return encoded

    def _build_sequential_model(self, input_dim, output_dim):
        layers = []
        previous_dim = input_dim
        activation_cls = self._get_activation()
        for hidden_dim in self.hidden_layer_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if activation_cls is not None:
                layers.append(activation_cls())
            previous_dim = hidden_dim
        layers.append(nn.Linear(previous_dim, output_dim))
        return nn.Sequential(*layers)

    def _build_balance(self, num_layers):
        partitions = int(self.n_workers)
        if partitions < 2:
            raise ValueError("GPipeMLP requires at least 2 workers/devices.")
        if num_layers < partitions:
            raise ValueError(
                f"GPipeMLP requires at least one layer per partition. Got {num_layers} layers for {partitions} workers."
            )

        base = num_layers // partitions
        remainder = num_layers % partitions
        return [base + (1 if index < remainder else 0) for index in range(partitions)]

    def _build_model(self, input_dim, output_dim):
        if not torch.cuda.is_available():
            raise RuntimeError("GPipeMLP requires CUDA, but CUDA is not available.")
        if torch.cuda.device_count() < int(self.n_workers):
            raise RuntimeError(
                f"GPipeMLP requires {self.n_workers} CUDA devices, but only {torch.cuda.device_count()} are available."
            )
        if int(self.chunks) < int(self.n_workers):
            raise ValueError("GPipeMLP requires chunks >= n_workers for a meaningful pipeline.")

        sequential_model = self._build_sequential_model(input_dim, output_dim)
        balance = self._build_balance(len(sequential_model))
        self.partition_devices_ = [torch.device(f"cuda:{index}") for index in range(int(self.n_workers))]
        self.input_device_ = self.partition_devices_[0]
        gpipe_kwargs = {
            "balance": balance,
            "chunks": int(self.chunks),
            "checkpoint": self.checkpoint,
            "devices": self.partition_devices_,
        }
        valid_parameters = inspect.signature(GPipe.__init__).parameters
        filtered_kwargs = {key: value for key, value in gpipe_kwargs.items() if key in valid_parameters}
        self.model_ = GPipe(sequential_model, **filtered_kwargs)
        self.balance_ = balance

    def _build_optimizer(self):
        if self.solver == "adam":
            return torch.optim.Adam(
                self.model_.parameters(),
                lr=self.learning_rate_init,
                weight_decay=self.alpha,
            )
        if self.solver == "sgd":
            return torch.optim.SGD(
                self.model_.parameters(),
                lr=self.learning_rate_init,
                momentum=self.momentum,
                weight_decay=self.alpha,
            )
        raise ValueError(f"Unsupported solver '{self.solver}'. Use 'adam' or 'sgd'.")

    def _create_loader(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self._is_binary:
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long)

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        return DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self._get_batch_size(len(X)),
            shuffle=self.shuffle,
            generator=generator,
            pin_memory=True,
        )

    def _evaluate_loss(self, X, y, criterion):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.input_device_)
        if self._is_binary:
            y_tensor = torch.tensor(y, dtype=torch.float32, device=self.partition_devices_[-1]).unsqueeze(1)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.partition_devices_[-1])

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            loss = criterion(logits, y_tensor)
        self.model_.train()
        return float(loss.item())

    def fit(self, X, y):
        self._require_gpipe()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        torch.manual_seed(self.random_state)
        torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        self.n_features_in_ = X.shape[1]
        y_encoded = self._encode_targets(y)

        X_train = X
        y_train = y_encoded
        X_val = None
        y_val = None

        if self.early_stopping and len(X) > 1:
            X_train, X_val, y_train, y_val = train_test_split(
                X,
                y_encoded,
                test_size=self.validation_fraction,
                random_state=self.random_state,
                stratify=y_encoded if len(self.classes_) > 1 else None,
            )

        output_dim = 1 if self._is_binary else len(self.classes_)
        self._build_model(self.n_features_in_, output_dim)
        optimizer = self._build_optimizer()
        criterion = nn.BCEWithLogitsLoss() if self._is_binary else nn.CrossEntropyLoss()
        train_loader = self._create_loader(X_train, y_train)

        best_state = copy.deepcopy(self.model_.state_dict())
        best_loss = float("inf")
        stale_epochs = 0

        for epoch in range(self.max_iter):
            self.model_.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.input_device_, non_blocking=True)
                y_batch = y_batch.to(self.partition_devices_[-1], non_blocking=True)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            monitored_loss = epoch_loss / len(train_loader.dataset)
            if X_val is not None and y_val is not None:
                monitored_loss = self._evaluate_loss(X_val, y_val, criterion)

            if best_loss - monitored_loss > self.tol:
                best_loss = monitored_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if self.verbose:
                print(f"Epoch {epoch + 1}/{self.max_iter} - loss={monitored_loss:.6f}")

            if self.early_stopping and stale_epochs >= self.n_iter_no_change:
                break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.n_iter_ = epoch + 1
        self.execution_mode_ = "gpipe"
        return self

    def _get_logits(self, X):
        self._require_gpipe()
        X_tensor = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32, device=self.input_device_)
        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
        return logits.detach().cpu().numpy()

    def predict(self, X):
        logits = self._get_logits(X)
        if self._is_binary:
            indices = (logits.ravel() >= 0).astype(int)
        else:
            indices = np.argmax(logits, axis=1)
        return self.classes_[indices]

    def predict_proba(self, X):
        logits = self._get_logits(X)
        if self._is_binary:
            probabilities = torch.sigmoid(torch.tensor(logits)).numpy().ravel()
            return np.column_stack([1.0 - probabilities, probabilities])
        return torch.softmax(torch.tensor(logits), dim=1).numpy()

    def decision_function(self, X):
        logits = self._get_logits(X)
        if self._is_binary:
            return logits.ravel()
        return logits
