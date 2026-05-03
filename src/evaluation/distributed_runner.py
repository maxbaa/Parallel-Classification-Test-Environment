from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DistributedTrainingResult:
    metrics: dict[str, float | None]
    execution_mode: str


DISTRIBUTED_ALGORITHMS = {"FSDPMLP", "DualPipeClassifier"}


def requires_distributed_execution(algorithm_name: str, algorithm_params: dict[str, Any]) -> bool:
    return algorithm_name in DISTRIBUTED_ALGORITHMS and int(algorithm_params.get("n_workers", 1)) > 1


def run_distributed_training(
    algorithm_name: str,
    algorithm_params: dict[str, Any],
    X_train: Any,
    X_test: Any,
    y_train: Any,
    y_test: Any,
    workdir: Path,
) -> DistributedTrainingResult:
    with tempfile.TemporaryDirectory(prefix="pcte_dist_", dir=workdir) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        data_path = temp_dir / "dataset.npz"
        request_path = temp_dir / "request.json"
        output_path = temp_dir / "output.json"

        np.savez_compressed(
            data_path,
            X_train=np.asarray(X_train),
            X_test=np.asarray(X_test),
            y_train=np.asarray(y_train),
            y_test=np.asarray(y_test),
        )

        request = {
            "algorithm_name": algorithm_name,
            "algorithm_params": algorithm_params,
            "data_path": str(data_path),
            "output_path": str(output_path),
        }
        request_path.write_text(json.dumps(request), encoding="utf-8")

        env = os.environ.copy()
        src_path = str((workdir / "src").resolve())
        env["PYTHONPATH"] = src_path if not env.get("PYTHONPATH") else src_path + os.pathsep + env["PYTHONPATH"]

        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={int(algorithm_params['n_workers'])}",
            "-m",
            "evaluation.distributed_entrypoint",
            "--request",
            str(request_path),
        ]
        subprocess.run(
            command,
            check=True,
            cwd=str(workdir),
            env=env,
        )

        result = json.loads(output_path.read_text(encoding="utf-8"))
        execution_mode = str(result.pop("execution_mode", "distributed"))
        return DistributedTrainingResult(metrics=result, execution_mode=execution_mode)
