from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from catalog import (
    build_algorithm_configs,
    build_dataset_configs,
    build_scenario_configs,
)
from evaluation.configs import ExperimentDefinition
from evaluation.persistence import (
    create_run_directory,
    save_config_copy,
    save_results_bundle,
)
from evaluation.reporting import build_summary, generate_report_plots, write_summary_markdown
from evaluation.runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run classification benchmarks and persist structured results.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/breast_cancer.yaml"),
        help="Path to the YAML experiment configuration.",
    )
    return parser.parse_args()


def load_yaml_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if not isinstance(config, dict):
        raise ValueError("Experiment configuration must be a YAML mapping at the top level.")

    return config


def build_experiment_definition(config: dict[str, Any]) -> ExperimentDefinition:
    return ExperimentDefinition(
        run_name=str(config.get("run_name", "benchmark_run")),
        repetitions=int(config.get("repetitions", 1)),
        output_dir=str(config.get("output_dir", "results")),
        enable_gpu_monitoring=bool(config.get("enable_gpu_monitoring", True)),
        continue_on_error=bool(config.get("continue_on_error", True)),
        algorithms=build_algorithm_configs(config.get("algorithms", [])),
        datasets=build_dataset_configs(config.get("datasets", [])),
        scenarios=build_scenario_configs(config.get("scenarios", [])),
    )


def main() -> None:
    args = parse_args()
    raw_config = load_yaml_config(args.config)
    experiment = build_experiment_definition(raw_config)

    run_dir = create_run_directory(
        run_name=experiment.run_name,
        output_root=Path(experiment.output_dir),
    )

    save_config_copy(args.config, raw_config, run_dir)

    runner = ExperimentRunner(
        algorithms=experiment.algorithms,
        datasets=experiment.datasets,
        scenarios=experiment.scenarios,
        repetitions=experiment.repetitions,
        enable_gpu_monitoring=experiment.enable_gpu_monitoring,
        continue_on_error=experiment.continue_on_error,
    )

    results_df = runner.run()
    summary_df = build_summary(results_df)
    plot_paths = generate_report_plots(results_df, summary_df, run_dir / "plots")
    markdown_path = write_summary_markdown(summary_df, run_dir / "summary")

    metadata = {
        "run_name": experiment.run_name,
        "config_path": str(args.config),
        "plots": [str(path) for path in plot_paths],
        "summary_markdown": str(markdown_path),
        "repetitions": experiment.repetitions,
        "continue_on_error": experiment.continue_on_error,
        "enable_gpu_monitoring": experiment.enable_gpu_monitoring,
    }

    save_results_bundle(
        run_dir=run_dir,
        results_df=results_df,
        summary_df=summary_df,
        metadata=metadata,
        failures=runner.failures,
    )

    print(json.dumps({"run_dir": str(run_dir)}, indent=2))


if __name__ == "__main__":
    main()
