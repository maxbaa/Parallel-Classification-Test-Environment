import copy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import train_test_split

try:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.utils.data.distributed import DistributedSampler
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.utils.data import DataLoader, TensorDataset
    _TORCH_IMPORT_ERROR = None
except ImportError as exc:
    torch = None
    dist = None
    nn = None
    FSDP = None
    DataLoader = None
    DistributedSampler = None
    TensorDataset = None
    _TORCH_IMPORT_ERROR = exc


class _TorchMLP(nn.Module if nn is not None else object):
    def __init__(self, input_dim, hidden_layer_sizes, activation, output_dim):
        super().__init__()

        layers = []
        previous_dim = input_dim
        for hidden_dim in hidden_layer_sizes:
            layers.append(nn.Linear(previous_dim, hidden_dim))
            if activation is not None:
                layers.append(activation())
            previous_dim = hidden_dim

        layers.append(nn.Linear(previous_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, X):
        return self.network(X)


class FSDPMLP(ClassifierMixin, BaseEstimator):
    """PyTorch MLP that requires true FSDP execution when configured."""

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
        device=None,
        n_workers=2,
        use_fsdp=True,
        sync_module_states=True,
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
        self.device = device
        self.n_workers = n_workers
        self.use_fsdp = use_fsdp
        self.sync_module_states = sync_module_states

    def _require_torch(self):
        if _TORCH_IMPORT_ERROR is not None:
            raise ImportError(
                "PyTorch is required for FSDPMLP. Install project dependencies "
                "again so that 'torch' is available."
            ) from _TORCH_IMPORT_ERROR

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

    def _get_device(self):
        if self.device is not None:
            return torch.device(self.device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _fsdp_ready(self):
        if not self.use_fsdp or FSDP is None:
            return False
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return False
        world_size = dist.get_world_size()
        return (
            self.device_.type == "cuda"
            and world_size >= 2
            and self.n_workers == world_size
        )

    def _get_batch_size(self, n_samples):
        if self.batch_size == "auto":
            return min(200, n_samples)
        return max(1, int(self.batch_size))

    def _encode_targets(self, y):
        self.classes_, encoded = np.unique(y, return_inverse=True)
        self._is_binary = len(self.classes_) == 2
        return encoded

    def _build_model(self, input_dim, output_dim):
        model = _TorchMLP(
            input_dim=input_dim,
            hidden_layer_sizes=self.hidden_layer_sizes,
            activation=self._get_activation(),
            output_dim=output_dim,
        ).to(self.device_)

        if self.use_fsdp:
            if not self._fsdp_ready():
                raise RuntimeError(
                    "FSDPMLP is configured to use FSDP, but the distributed CUDA/FSDP runtime is not ready."
                )
            self.model_ = FSDP(
                model,
                device_id=torch.cuda.current_device(),
                sync_module_states=self.sync_module_states,
            )
            self.execution_mode_ = "fsdp"
            return

        self.model_ = model
        self.execution_mode_ = "local"

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
        raise ValueError(
            f"Unsupported solver '{self.solver}'. Use 'adam' or 'sgd'."
        )

    def _create_loader(self, X, y):
        X_tensor = torch.tensor(X, dtype=torch.float32)
        if self._is_binary:
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long)

        generator = torch.Generator()
        generator.manual_seed(self.random_state)

        sampler = None
        if self._fsdp_ready() and DistributedSampler is not None:
            sampler = DistributedSampler(
                TensorDataset(X_tensor, y_tensor),
                num_replicas=dist.get_world_size(),
                rank=dist.get_rank(),
                shuffle=self.shuffle,
                seed=self.random_state,
            )

        return DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=self._get_batch_size(len(X)),
            shuffle=self.shuffle if sampler is None else False,
            sampler=sampler,
            generator=generator,
            pin_memory=self.device_.type == "cuda",
        ), sampler

    def _evaluate_loss(self, X, y, criterion):
        X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device_)
        if self._is_binary:
            y_tensor = torch.tensor(
                y, dtype=torch.float32, device=self.device_
            ).unsqueeze(1)
        else:
            y_tensor = torch.tensor(y, dtype=torch.long, device=self.device_)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            loss = criterion(logits, y_tensor)
        self.model_.train()
        return float(loss.item())

    def fit(self, X, y):
        self._require_torch()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)

        torch.manual_seed(self.random_state)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)
        np.random.seed(self.random_state)

        self.device_ = self._get_device()
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
        criterion = (
            nn.BCEWithLogitsLoss() if self._is_binary else nn.CrossEntropyLoss()
        )
        train_loader, train_sampler = self._create_loader(X_train, y_train)

        best_state = copy.deepcopy(self.model_.state_dict())
        best_loss = float("inf")
        stale_epochs = 0

        for epoch in range(self.max_iter):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            self.model_.train()
            epoch_loss = 0.0

            for X_batch, y_batch in train_loader:
                if self.device_.type == "cuda":
                    X_batch = X_batch.to(self.device_, non_blocking=True)
                    y_batch = y_batch.to(self.device_, non_blocking=True)
                else:
                    X_batch = X_batch.to(self.device_)
                    y_batch = y_batch.to(self.device_)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model_(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(X_batch)

            epoch_loss_tensor = torch.tensor(
                [epoch_loss, float(len(train_loader.dataset))],
                dtype=torch.float64,
                device=self.device_,
            )
            if self._fsdp_ready():
                dist.all_reduce(epoch_loss_tensor, op=dist.ReduceOp.SUM)
            monitored_loss = epoch_loss_tensor[0].item() / max(epoch_loss_tensor[1].item(), 1.0)
            if X_val is not None and y_val is not None:
                monitored_loss = self._evaluate_loss(X_val, y_val, criterion)

            if best_loss - monitored_loss > self.tol:
                best_loss = monitored_loss
                best_state = copy.deepcopy(self.model_.state_dict())
                stale_epochs = 0
            else:
                stale_epochs += 1

            if self.verbose:
                print(
                    f"Epoch {epoch + 1}/{self.max_iter} - loss={monitored_loss:.6f}"
                )

            if self.early_stopping and stale_epochs >= self.n_iter_no_change:
                break

        self.model_.load_state_dict(best_state)
        self.model_.eval()
        self.n_iter_ = epoch + 1
        return self

    def _get_logits(self, X):
        self._require_torch()
        X_tensor = torch.tensor(
            np.asarray(X, dtype=np.float32),
            dtype=torch.float32,
            device=self.device_,
        )
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
