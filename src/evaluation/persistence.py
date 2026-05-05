from __future__ import annotations

import json
import platform
import socket
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    import torch
except ImportError:
    torch = None


def _slugify(value: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    compact = "_".join(part for part in sanitized.split("_") if part)
    return compact or "run"


def create_run_directory(run_name: str, output_root: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{timestamp}_{_slugify(run_name)}"

    for subdir in ("raw", "summary", "plots", "logs", "config"):
        (run_dir / subdir).mkdir(parents=True, exist_ok=True)

    return run_dir


def save_config_copy(source_path: Path, raw_config: dict[str, Any], run_dir: Path) -> None:
    config_dir = run_dir / "config"
    with (config_dir / source_path.name).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(raw_config, handle, sort_keys=False)


def _build_system_metadata() -> dict[str, Any]:
    gpu_info: dict[str, Any] = {
        "torch_available": torch is not None,
        "cuda_available": False,
        "device_count": 0,
        "device_names": [],
    }

    if torch is not None and torch.cuda.is_available():
        gpu_info = {
            "torch_available": True,
            "cuda_available": True,
            "device_count": torch.cuda.device_count(),
            "device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        }

    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu": gpu_info,
    }


def save_results_bundle(
    run_dir: Path,
    results_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    metadata: dict[str, Any],
    failures: list[dict[str, Any]],
) -> None:
    results_df.to_csv(run_dir / "raw" / "runs.csv", index=False)
    summary_df.to_csv(run_dir / "summary" / "aggregated_metrics.csv", index=False)

    if "confusion_matrix_json" in results_df.columns:
        confusion_records: list[dict[str, Any]] = []
        for row in results_df.itertuples(index=False):
            payload = getattr(row, "confusion_matrix_json", "")
            if isinstance(payload, str) and payload.strip():
                confusion_records.append(
                    {
                        "algorithm": row.algorithm,
                        "dataset": row.dataset,
                        "scenario": row.scenario,
                        "run_index": row.run_index,
                        "status": row.status,
                        "confusion_matrix": json.loads(payload),
                    }
                )
        with (run_dir / "raw" / "confusion_matrices.json").open("w", encoding="utf-8") as handle:
            json.dump(confusion_records, handle, indent=2)

    if "classification_report_json" in results_df.columns:
        report_records: list[dict[str, Any]] = []
        for row in results_df.itertuples(index=False):
            payload = getattr(row, "classification_report_json", "")
            if isinstance(payload, str) and payload.strip():
                report_records.append(
                    {
                        "algorithm": row.algorithm,
                        "dataset": row.dataset,
                        "scenario": row.scenario,
                        "run_index": row.run_index,
                        "status": row.status,
                        "classification_report": json.loads(payload),
                    }
                )
        with (run_dir / "raw" / "classification_reports.json").open("w", encoding="utf-8") as handle:
            json.dump(report_records, handle, indent=2)

    with (run_dir / "logs" / "failures.json").open("w", encoding="utf-8") as handle:
        json.dump(failures, handle, indent=2)

    combined_metadata = {
        **metadata,
        "created_at": datetime.now().isoformat(),
        "system": _build_system_metadata(),
    }
    with (run_dir / "summary" / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(combined_metadata, handle, indent=2)
