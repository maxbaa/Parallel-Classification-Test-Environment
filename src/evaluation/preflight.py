from __future__ import annotations

import importlib
from typing import Any

from evaluation.configs import AlgorithmConfig, ScenarioConfig


def _import_required(module_name: str, reason: str) -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception as exc:
        raise RuntimeError(f"{reason}: required module '{module_name}' is not available.") from exc


def _require_torch_cuda(reason: str):
    torch = _import_required("torch", reason)
    if not torch.cuda.is_available():
        raise RuntimeError(f"{reason}: CUDA is not available in the current PyTorch runtime.")
    return torch


def _require_cuda_device_count(minimum: int, reason: str):
    torch = _require_torch_cuda(reason)
    available = int(torch.cuda.device_count())
    if available < int(minimum):
        raise RuntimeError(
            f"{reason}: requires at least {minimum} CUDA devices, but only {available} are available."
        )
    return torch


def _validate_algorithm_runtime(algorithm_name: str, algorithm_params: dict[str, Any], scenario_name: str) -> None:
    reason = f"Preflight check failed for {algorithm_name} in scenario '{scenario_name}'"

    if algorithm_name == "ThunderSVC":
        _require_torch_cuda(reason)
        _import_required("thundersvm", reason)
        return

    if algorithm_name in {"CuMLKNN", "CuMLRF"}:
        _require_torch_cuda(reason)
        _import_required("cuml", reason)
        return

    if algorithm_name == "CuMLMultiGPUKNN":
        required_workers = int(algorithm_params.get("n_workers", 2))
        _require_cuda_device_count(required_workers, reason)
        _import_required("cuml.dask", reason)
        _import_required("dask_cuda", reason)
        _import_required("distributed", reason)
        return

    if algorithm_name == "GPipeMLP":
        required_workers = int(algorithm_params.get("n_workers", 2))
        _require_cuda_device_count(required_workers, reason)
        _import_required("torchgpipe", reason)
        chunks = int(algorithm_params.get("chunks", 1))
        if chunks < required_workers:
            raise RuntimeError(
                f"{reason}: requires chunks >= n_workers, but got chunks={chunks} and n_workers={required_workers}."
            )
        return

    if algorithm_name == "FSDPMLP":
        required_workers = int(algorithm_params.get("n_workers", 1))
        if not bool(algorithm_params.get("use_fsdp", True)):
            raise RuntimeError(f"{reason}: use_fsdp must stay enabled for FSDPMLP.")
        _require_cuda_device_count(required_workers, reason)
        _import_required("torch.distributed.fsdp", reason)
        return

    if algorithm_name == "BaselineMLP":
        device = str(algorithm_params.get("device", "")).lower()
        if device == "cuda":
            _require_torch_cuda(reason)


def validate_experiment_environment(
    algorithms: list[AlgorithmConfig],
    scenarios: list[ScenarioConfig],
) -> None:
    scenario_by_name = {scenario.name: scenario for scenario in scenarios}

    for algorithm in algorithms:
        candidate_scenarios = algorithm.allowed_scenarios or list(scenario_by_name.keys())
        for scenario_name in candidate_scenarios:
            scenario = scenario_by_name.get(scenario_name)
            if scenario is None:
                raise RuntimeError(
                    f"Preflight check failed for {algorithm.name}: unknown configured scenario '{scenario_name}'."
                )
            merged_params = {
                **algorithm.parameters,
                **scenario.params,
            }
            _validate_algorithm_runtime(algorithm.name, merged_params, scenario_name)
