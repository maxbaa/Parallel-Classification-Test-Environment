from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


RUNTIME_GROUPS: dict[str, list[str]] = {
    "core": [
        "BaselineSVC",
        "CascadeSVC",
        "BaselineRF",
        "HybridRF",
        "BaselineKNN",
        "BaselineMLP",
        "GPipeMLP",
        "FSDPMLP",
    ],
    "rapids": [
        "CuMLRF",
        "CuMLKNN",
        "CuMLMultiGPUKNN",
    ],
    "thunder": [
        "ThunderSVC",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one benchmark config with a filtered runtime-specific algorithm set.",
    )
    parser.add_argument("--runtime", choices=sorted(RUNTIME_GROUPS), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=None)
    parser.add_argument("--output-config", type=Path, default=None)
    parser.add_argument("--print-only", action="store_true")
    return parser.parse_args()


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config {path} must contain a YAML mapping.")
    return data


def filter_config(config: dict, runtime: str) -> dict:
    allowed = set(RUNTIME_GROUPS[runtime])
    algorithms = [entry for entry in config.get("algorithms", []) if entry.get("name") in allowed]
    if not algorithms:
        raise ValueError(f"No algorithms from runtime '{runtime}' found in config.")

    filtered = dict(config)
    filtered["algorithms"] = algorithms
    filtered["run_name"] = f"{config.get('run_name', 'benchmark')}_{runtime}"
    return filtered


def write_config(config: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)


def main() -> None:
    args = parse_args()
    base_config = load_config(args.config)
    filtered_config = filter_config(base_config, args.runtime)

    temp_dir: tempfile.TemporaryDirectory[str] | None = None
    if args.output_config is not None:
        output_path = args.output_config
        write_config(filtered_config, output_path)
    else:
        temp_dir = tempfile.TemporaryDirectory(prefix="pcte_runtime_")
        output_path = Path(temp_dir.name) / f"{args.config.stem}_{args.runtime}.yaml"
        write_config(filtered_config, output_path)

    if args.print_only:
        print(output_path)
        return

    python_executable = str(args.python) if args.python is not None else sys.executable
    command = [python_executable, "-m", "pcte_cli", "--config", str(output_path)]
    try:
        raise SystemExit(subprocess.call(command))
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


if __name__ == "__main__":
    main()
