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
        else:
            raise ValueError(f"Unsupported distributed algorithm '{algorithm_name}'.")

        if dist.get_rank() == 0:
            Path(request["output_path"]).write_text(json.dumps(result), encoding="utf-8")
    finally:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
