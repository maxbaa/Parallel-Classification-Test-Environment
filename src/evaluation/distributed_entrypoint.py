from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def _init_torch():
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch, dist, local_rank


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray | None) -> dict[str, float | None]:
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = np.unique(y_true)
    is_binary = len(labels) == 2

    roc_auc = None
    if y_score is not None:
        try:
            if is_binary:
                roc_auc = float(roc_auc_score(y_true, y_score))
            else:
                roc_auc = float(roc_auc_score(y_true, y_score, multi_class="ovr", average="weighted"))
        except Exception:
            roc_auc = None

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "roc_auc": roc_auc,
    }


def _run_fsdp(request: dict[str, object], X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    from algorithms.sklearn_wrappers import create_fsdp_mlp

    params = dict(request["algorithm_params"])
    model = create_fsdp_mlp(params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)
        y_score = y_proba[:, 1] if y_proba.ndim == 2 and y_proba.shape[1] == 2 else y_proba
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)

    result = _compute_metrics(y_test, y_pred, y_score)
    result["execution_mode"] = getattr(model, "execution_mode_", "fsdp")
    return result


def _run_dualpipe_fallback(request: dict[str, object], X_train: np.ndarray, X_test: np.ndarray, y_train: np.ndarray, y_test: np.ndarray) -> dict[str, object]:
    import torch
    import torch.distributed as dist
    from torch import nn
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, TensorDataset
    from torch.utils.data.distributed import DistributedSampler

    from algorithms.sklearn_wrappers import _load_dualpipe_classifier

    dualpipe_module = _load_dualpipe_classifier().__module__
    fallback_module = __import__(dualpipe_module, fromlist=["_DualPipeFallbackMLP"])
    FallbackMLP = getattr(fallback_module, "_DualPipeFallbackMLP")

    params = dict(request["algorithm_params"])
    activation_name = params.get("activation", "relu")
    hidden_layer_sizes = tuple(params.get("hidden_layer_sizes", (100,)))
    batch_size = params.get("batch_size", "auto")
    learning_rate_init = float(params.get("learning_rate_init", 1e-3))
    alpha = float(params.get("alpha", 1e-4))
    solver = params.get("solver", "adam")
    max_iter = int(params.get("max_iter", 200))
    shuffle = bool(params.get("shuffle", True))
    momentum = float(params.get("momentum", 0.9))
    random_state = int(params.get("random_state", 42))
    tol = float(params.get("tol", 1e-4))
    verbose = bool(params.get("verbose", False))

    torch.manual_seed(random_state)
    torch.cuda.manual_seed_all(random_state)
    np.random.seed(random_state)

    device = torch.device(f"cuda:{torch.cuda.current_device()}")
    classes, y_encoded = np.unique(y_train, return_inverse=True)
    is_binary = len(classes) == 2

    activations = {
        "identity": None,
        "logistic": nn.Sigmoid,
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
    }
    if activation_name not in activations:
        raise ValueError(f"Unsupported activation '{activation_name}'.")

    output_dim = 1 if is_binary else len(classes)
    model = FallbackMLP(
        input_dim=X_train.shape[1],
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activations[activation_name],
        output_dim=output_dim,
    ).to(device)
    ddp_model = DDP(model, device_ids=[torch.cuda.current_device()])

    if solver == "adam":
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=learning_rate_init, weight_decay=alpha)
    elif solver == "sgd":
        optimizer = torch.optim.SGD(ddp_model.parameters(), lr=learning_rate_init, momentum=momentum, weight_decay=alpha)
    else:
        raise ValueError(f"Unsupported solver '{solver}'. Use 'adam' or 'sgd'.")

    criterion = nn.BCEWithLogitsLoss() if is_binary else nn.CrossEntropyLoss()
    x_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.float32 if is_binary else torch.long)
    if is_binary:
        y_tensor = y_tensor.unsqueeze(1)
    dataset = TensorDataset(x_tensor, y_tensor)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=shuffle,
        seed=random_state,
    )

    effective_batch_size = min(200, len(X_train)) if batch_size == "auto" else max(1, int(batch_size))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        sampler=sampler,
        pin_memory=True,
    )

    best_loss = float("inf")
    best_state = None
    stale_epochs = 0

    for epoch in range(max_iter):
        sampler.set_epoch(epoch)
        ddp_model.train()
        epoch_loss = 0.0

        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = ddp_model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(X_batch)

        loss_tensor = torch.tensor([epoch_loss, float(len(dataset))], dtype=torch.float64, device=device)
        dist.all_reduce(loss_tensor, op=dist.ReduceOp.SUM)
        monitored_loss = loss_tensor[0].item() / max(loss_tensor[1].item(), 1.0)

        if best_loss - monitored_loss > tol:
            best_loss = monitored_loss
            best_state = {k: v.detach().cpu() for k, v in ddp_model.module.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1

        if verbose and dist.get_rank() == 0:
            print(f"Epoch {epoch + 1}/{max_iter} - loss={monitored_loss:.6f}")

    if best_state is not None:
        ddp_model.module.load_state_dict(best_state)

    ddp_model.eval()
    with torch.no_grad():
        logits = ddp_model.module(torch.tensor(X_test, dtype=torch.float32, device=device))
    logits_np = logits.detach().cpu().numpy()
    if is_binary:
        y_pred = classes[(logits_np.ravel() >= 0).astype(int)]
        probabilities = torch.sigmoid(torch.tensor(logits_np)).numpy().ravel()
        y_score = probabilities
    else:
        indices = np.argmax(logits_np, axis=1)
        y_pred = classes[indices]
        y_score = torch.softmax(torch.tensor(logits_np), dim=1).numpy()

    result = _compute_metrics(y_test, y_pred, y_score)
    result["execution_mode"] = "ddp_fallback"
    return result


def main() -> None:
    import argparse
    import torch.distributed as dist

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    args = parser.parse_args()

    _init_torch()

    request_path = Path(args.request)
    request = json.loads(request_path.read_text(encoding="utf-8"))
    data = np.load(request["data_path"])
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]

    result: dict[str, object] | None = None
    try:
        algorithm_name = str(request["algorithm_name"])
        if algorithm_name == "FSDPMLP":
            result = _run_fsdp(request, X_train, X_test, y_train, y_test)
        elif algorithm_name == "DualPipeClassifier":
            result = _run_dualpipe_fallback(request, X_train, X_test, y_train, y_test)
        else:
            raise ValueError(f"Unsupported distributed algorithm '{algorithm_name}'.")

        if dist.get_rank() == 0:
            Path(request["output_path"]).write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
