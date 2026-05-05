from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from evaluation.metrics import compute_quality_metrics


def _init_torch():
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl")
    return torch, dist, local_rank


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

    metrics = compute_quality_metrics(model, X_test, y_test)
    result = {
        "accuracy": float(metrics.accuracy),
        "balanced_accuracy": float(metrics.balanced_accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "roc_auc": metrics.roc_auc,
        "confusion_matrix_json": metrics.confusion_matrix_json,
        "classification_report_json": metrics.classification_report_json,
    }
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
