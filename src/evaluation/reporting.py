from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "train_time_sec",
    "time_per_sample_sec",
    "cpu_avg",
    "cpu_max",
    "ram_avg_gb",
    "ram_max_gb",
    "gpu_max_gb",
]


def _context_label(row: pd.Series) -> str:
    dataset_group = str(row.get("dataset_group", row.get("dataset", "")))
    split_label = str(row.get("split_label", "") or "").strip()
    scenario = str(row.get("scenario", ""))
    dataset_label = dataset_group if not split_label else f"{dataset_group} ({split_label})"
    return dataset_label if not scenario else f"{dataset_label} | {scenario}"


def _slugify(value: str) -> str:
    sanitized = "".join(char.lower() if char.isalnum() else "_" for char in value.strip())
    compact = "_".join(part for part in sanitized.split("_") if part)
    return compact or "plot"


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    grouped = (
        results_df.assign(success_flag=results_df["status"].eq("success").astype(int))
        .groupby(["dataset", "dataset_group", "split_label", "scenario", "algorithm"], dropna=False)
        .agg(
            runs_total=("run_index", "count"),
            runs_success=("success_flag", "sum"),
            n_train_samples_mean=("n_train_samples", "mean"),
            n_test_samples_mean=("n_test_samples", "mean"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
            balanced_accuracy_mean=("balanced_accuracy", "mean"),
            precision_mean=("precision", "mean"),
            recall_mean=("recall", "mean"),
            f1_mean=("f1", "mean"),
            roc_auc_mean=("roc_auc", "mean"),
            train_time_sec_mean=("train_time_sec", "mean"),
            time_per_sample_sec_mean=("time_per_sample_sec", "mean"),
            cpu_avg_mean=("cpu_avg", "mean"),
            ram_max_gb_mean=("ram_max_gb", "mean"),
            gpu_max_gb_mean=("gpu_max_gb", "mean"),
            best_cv_score_mean=("best_cv_score", "mean"),
            evaluated_candidates_mean=("evaluated_candidates", "mean"),
        )
        .reset_index()
    )
    grouped["success_rate"] = grouped["runs_success"] / grouped["runs_total"]
    return grouped.sort_values(
        ["dataset_group", "n_train_samples_mean", "scenario", "algorithm"]
    ).reset_index(drop=True)


def build_failure_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty or "status" not in results_df.columns:
        return pd.DataFrame()

    failed_df = results_df.loc[results_df["status"].eq("failed")].copy()
    if failed_df.empty:
        return pd.DataFrame(
            columns=["dataset", "scenario", "algorithm", "error_type", "error_message", "failure_count"]
        )

    grouped = (
        failed_df.groupby(
            ["dataset", "scenario", "algorithm", "error_type", "error_message"],
            dropna=False,
        )
        .size()
        .reset_index(name="failure_count")
        .sort_values(["failure_count", "dataset", "scenario", "algorithm"], ascending=[False, True, True, True])
        .reset_index(drop=True)
    )
    return grouped


def _plot_metric(summary_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> Path | None:
    if summary_df.empty or metric not in summary_df.columns:
        return None

    plot_df = summary_df.copy()
    plot_df["context"] = plot_df.apply(_context_label, axis=1)
    pivot_df = plot_df.pivot(index="algorithm", columns="context", values=metric)
    if pivot_df.empty:
        return None

    ax = pivot_df.plot(kind="bar", figsize=(12, 6))
    ax.set_title(ylabel)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Algorithm")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(title="Dataset | Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def _plot_heatmap(summary_df: pd.DataFrame, metric: str, title: str, output_path: Path, fmt: str = ".3f") -> Path | None:
    if summary_df.empty or metric not in summary_df.columns:
        return None

    plot_df = summary_df.copy()
    plot_df["context"] = plot_df.apply(_context_label, axis=1)
    pivot_df = plot_df.pivot(index="algorithm", columns="context", values=metric)
    if pivot_df.empty:
        return None

    values = pivot_df.to_numpy(dtype=float)
    fig_width = max(8, 1.3 * len(pivot_df.columns))
    fig_height = max(5, 0.5 * len(pivot_df.index) + 2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    im = ax.imshow(values, aspect="auto", cmap="viridis")
    ax.set_title(title)
    ax.set_xticks(range(len(pivot_df.columns)))
    ax.set_xticklabels(pivot_df.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(pivot_df.index)))
    ax.set_yticklabels(pivot_df.index)

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            label = "NA" if np.isnan(value) else format(value, fmt)
            ax.text(col, row, label, ha="center", va="center", color="white", fontsize=8)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_tradeoff(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    required_columns = {"accuracy_mean", "train_time_sec_mean", "algorithm", "dataset", "scenario", "success_rate"}
    if summary_df.empty or not required_columns.issubset(summary_df.columns):
        return None

    plot_df = summary_df.dropna(subset=["accuracy_mean", "train_time_sec_mean"]).copy()
    if plot_df.empty:
        return None

    scenarios = list(plot_df["scenario"].dropna().unique())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(scenarios))))
    color_map = {scenario: colors[index] for index, scenario in enumerate(scenarios)}

    fig, ax = plt.subplots(figsize=(11, 7))
    for row in plot_df.itertuples(index=False):
        bubble_size = 120 + 240 * float(row.success_rate)
        ax.scatter(
            row.train_time_sec_mean,
            row.accuracy_mean,
            s=bubble_size,
            color=color_map.get(row.scenario, "#1f77b4"),
            alpha=0.8,
            edgecolors="black",
            linewidths=0.5,
        )
        ax.annotate(
            row.algorithm,
            (row.train_time_sec_mean, row.accuracy_mean),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )

    ax.set_xscale("log")
    ax.set_xlabel("Train Time Mean (s, log scale)")
    ax.set_ylabel("Accuracy Mean")
    ax.set_title("Accuracy vs. Train Time Trade-off")
    handles = [
        plt.Line2D([0], [0], marker="o", color="w", label=scenario, markerfacecolor=color_map[scenario], markersize=10)
        for scenario in scenarios
    ]
    if handles:
        ax.legend(handles=handles, title="Scenario", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(True, which="both", axis="both", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_resource_profiles(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    required = {"cpu_avg_mean", "ram_max_gb_mean", "gpu_max_gb_mean", "algorithm", "dataset", "scenario"}
    if summary_df.empty or not required.issubset(summary_df.columns):
        return None

    plot_df = summary_df.copy()
    plot_df["label"] = plot_df["algorithm"] + "\n" + plot_df["scenario"]
    if "split_label" in plot_df.columns:
        plot_df["label"] = plot_df["label"] + "\n" + plot_df["split_label"].fillna("")
    plot_df = plot_df.sort_values(["dataset_group", "n_train_samples_mean", "scenario", "algorithm"]).reset_index(drop=True)
    if plot_df.empty:
        return None

    positions = np.arange(len(plot_df))
    width = 0.26

    fig, ax = plt.subplots(figsize=(max(12, len(plot_df) * 0.8), 7))
    ax.bar(positions - width, plot_df["cpu_avg_mean"].fillna(0), width=width, label="CPU Avg (%)")
    ax.bar(positions, plot_df["ram_max_gb_mean"].fillna(0), width=width, label="RAM Peak (GB)")
    ax.bar(positions + width, plot_df["gpu_max_gb_mean"].fillna(0), width=width, label="GPU Peak (GB)")
    ax.set_xticks(positions)
    ax.set_xticklabels(plot_df["label"], rotation=30, ha="right")
    ax.set_title("Resource Profile per Algorithm and Scenario")
    ax.set_ylabel("Measured Value")
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_overview(summary_df: pd.DataFrame, output_path: Path) -> Path | None:
    if summary_df.empty:
        return None

    plot_df = summary_df.copy()
    plot_df["context"] = plot_df.apply(_context_label, axis=1)
    metrics = [
        ("accuracy_mean", "Accuracy"),
        ("f1_mean", "F1"),
        ("train_time_sec_mean", "Train Time (s)"),
        ("success_rate", "Success Rate"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes_flat = axes.flatten()

    for axis, (metric, title) in zip(axes_flat, metrics):
        pivot_df = plot_df.pivot(index="algorithm", columns="context", values=metric)
        pivot_df.plot(kind="bar", ax=axis)
        axis.set_title(title)
        axis.set_xlabel("Algorithm")
        axis.tick_params(axis="x", rotation=30)
        axis.legend().remove()

    handles, labels = axes_flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Dataset | Scenario", loc="upper right")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def _plot_train_scaling(results_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    required_columns = {"algorithm", "scenario", "n_train_samples", "train_time_sec", "status"}
    if results_df.empty or not required_columns.issubset(results_df.columns):
        return []

    dataset_column = "dataset_group" if "dataset_group" in results_df.columns else "dataset"
    plot_df = results_df.loc[
        results_df["status"].eq("success") & results_df["n_train_samples"].gt(0)
    ].copy()
    if plot_df.empty:
        return []

    grouped = (
        plot_df.groupby([dataset_column, "scenario", "algorithm", "n_train_samples"], dropna=False)
        .agg(
            train_time_sec_mean=("train_time_sec", "mean"),
            train_time_sec_std=("train_time_sec", "std"),
            runs=("run_index", "count"),
        )
        .reset_index()
        .sort_values([dataset_column, "scenario", "algorithm", "n_train_samples"])
    )

    output_paths: list[Path] = []
    for dataset_name, dataset_df in grouped.groupby(dataset_column, dropna=False):
        if dataset_df["n_train_samples"].nunique() < 2:
            continue

        fig, ax = plt.subplots(figsize=(10, 6))
        multi_scenario = dataset_df["scenario"].nunique() > 1
        for (scenario, algorithm), series_df in dataset_df.groupby(["scenario", "algorithm"], dropna=False):
            label = str(algorithm) if not multi_scenario else f"{algorithm} | {scenario}"
            ordered = series_df.sort_values("n_train_samples")
            ax.plot(
                ordered["n_train_samples"],
                ordered["train_time_sec_mean"],
                marker="o",
                linewidth=2,
                label=label,
            )

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Size of Training Set")
        ax.set_ylabel("Train Time Mean (s)")
        ax.set_title(f"Training Set Scaling: {dataset_name}")
        ax.grid(True, which="both", axis="both", alpha=0.25)
        ax.legend(
            title="Algorithm" if not multi_scenario else "Algorithm | Scenario",
            bbox_to_anchor=(1.02, 1),
            loc="upper left",
        )
        plt.tight_layout()

        output_path = output_dir / f"train_scaling_{_slugify(str(dataset_name))}.png"
        plt.savefig(output_path, dpi=180)
        plt.close(fig)
        output_paths.append(output_path)

    return output_paths


def generate_report_plots(results_df: pd.DataFrame, summary_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_paths = [
        _plot_overview(summary_df, output_dir / "overview.png"),
        _plot_metric(summary_df, "accuracy_mean", "Accuracy", output_dir / "accuracy.png"),
        _plot_metric(summary_df, "f1_mean", "F1 Score", output_dir / "f1_score.png"),
        _plot_metric(summary_df, "train_time_sec_mean", "Train Time (s)", output_dir / "train_time.png"),
        _plot_heatmap(summary_df, "success_rate", "Success Rate Heatmap", output_dir / "success_rate_heatmap.png", ".2%"),
        _plot_heatmap(summary_df, "accuracy_mean", "Accuracy Heatmap", output_dir / "accuracy_heatmap.png"),
        _plot_heatmap(summary_df, "train_time_sec_mean", "Train Time Heatmap", output_dir / "train_time_heatmap.png", ".2f"),
        _plot_tradeoff(summary_df, output_dir / "accuracy_vs_train_time.png"),
        _plot_resource_profiles(summary_df, output_dir / "resource_profiles.png"),
    ]
    plot_paths.extend(_plot_train_scaling(results_df, output_dir))

    if "gpu_max_gb_mean" in summary_df.columns and not summary_df["gpu_max_gb_mean"].fillna(0).eq(0).all():
        plot_paths.append(
            _plot_metric(summary_df, "gpu_max_gb_mean", "GPU Memory Peak (GB)", output_dir / "gpu_memory.png")
        )
    else:
        plot_paths.append(
            _plot_metric(summary_df, "ram_max_gb_mean", "RAM Peak (GB)", output_dir / "ram_peak.png")
        )

    return [path for path in plot_paths if path is not None]


def _write_metric_matrix(summary_df: pd.DataFrame, metric: str, output_path: Path) -> Path | None:
    if summary_df.empty or metric not in summary_df.columns:
        return None

    matrix_df = summary_df.copy()
    matrix_df["context"] = matrix_df.apply(_context_label, axis=1)
    pivot_df = matrix_df.pivot(index="algorithm", columns="context", values=metric)
    if pivot_df.empty:
        return None

    pivot_df.to_csv(output_path, encoding="utf-8")
    return output_path


def write_summary_tables(summary_df: pd.DataFrame, failures_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []

    if not summary_df.empty:
        leaderboard_df = summary_df.sort_values(
            ["success_rate", "accuracy_mean", "f1_mean", "train_time_sec_mean"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        leaderboard_path = output_dir / "leaderboard.csv"
        leaderboard_df.to_csv(leaderboard_path, index=False, encoding="utf-8")
        output_paths.append(leaderboard_path)

        for metric, filename in [
            ("accuracy_mean", "matrix_accuracy.csv"),
            ("balanced_accuracy_mean", "matrix_balanced_accuracy.csv"),
            ("f1_mean", "matrix_f1.csv"),
            ("train_time_sec_mean", "matrix_train_time.csv"),
            ("success_rate", "matrix_success_rate.csv"),
            ("gpu_max_gb_mean", "matrix_gpu_memory.csv"),
            ("ram_max_gb_mean", "matrix_ram_peak.csv"),
        ]:
            path = _write_metric_matrix(summary_df, metric, output_dir / filename)
            if path is not None:
                output_paths.append(path)

    failures_path = output_dir / "failure_summary.csv"
    failures_df.to_csv(failures_path, index=False, encoding="utf-8")
    output_paths.append(failures_path)
    return output_paths


def write_summary_markdown(summary_df: pd.DataFrame, failures_df: pd.DataFrame, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    markdown_path = output_dir / "summary.md"

    lines = [
        "# Experiment Summary",
        "",
    ]

    if summary_df.empty:
        lines.append("No successful runs were available for aggregation.")
    else:
        best_accuracy = summary_df.sort_values("accuracy_mean", ascending=False).head(5)
        highest_success = summary_df.sort_values("success_rate", ascending=False).head(5)
        fastest_success = summary_df.dropna(subset=["train_time_sec_mean"]).sort_values("train_time_sec_mean", ascending=True).head(5)
        lines.extend(
            [
                "## Top Accuracy",
                "",
                "| Dataset | Scenario | Algorithm | Accuracy | Balanced Acc. | F1 | CV Score | Train Time (s) | Success Rate |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in best_accuracy.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.scenario} | {row.algorithm} | "
                f"{row.accuracy_mean:.4f} | {row.balanced_accuracy_mean:.4f} | {row.f1_mean:.4f} | "
                f"{row.best_cv_score_mean:.4f} | {row.train_time_sec_mean:.4f} | {row.success_rate:.2%} |"
            )
        lines.extend(
            [
                "",
                "## Reliability Overview",
                "",
                "| Dataset | Scenario | Algorithm | Success Rate | Accuracy | F1 | Train Time (s) |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in highest_success.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.scenario} | {row.algorithm} | "
                f"{row.success_rate:.2%} | {row.accuracy_mean:.4f} | {row.f1_mean:.4f} | {row.train_time_sec_mean:.4f} |"
            )
        lines.extend(
            [
                "",
                "## Fastest Successful Runs",
                "",
                "| Dataset | Scenario | Algorithm | Train Time (s) | Accuracy | Success Rate |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in fastest_success.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.scenario} | {row.algorithm} | "
                f"{row.train_time_sec_mean:.4f} | {row.accuracy_mean:.4f} | {row.success_rate:.2%} |"
            )
        lines.extend(
            [
                "",
                "## Value Tables",
                "",
                "- `leaderboard.csv` contains the full sortable ranking with numeric metrics.",
                "- `matrix_accuracy.csv`, `matrix_balanced_accuracy.csv`, `matrix_f1.csv` and `matrix_success_rate.csv` contain comparison matrices by algorithm and dataset/scenario.",
                "- `matrix_train_time.csv`, `matrix_ram_peak.csv` and `matrix_gpu_memory.csv` contain numeric runtime/resource matrices.",
                "- `plots/train_scaling_*.png` shows how train time grows with training set size for every dataset that was configured with multiple train sizes.",
                "",
                "Missing bars in plots usually mean there was no successful run for that algorithm/scenario combination, so the chart had no numeric value to aggregate.",
            ]
        )

    lines.extend(["", "## Failure Analysis", ""])
    if failures_df.empty:
        lines.append("No failed runs were recorded.")
    else:
        lines.extend(
            [
                "| Dataset | Scenario | Algorithm | Error Type | Count | Error Message |",
                "| --- | --- | --- | --- | ---: | --- |",
            ]
        )
        for row in failures_df.head(20).itertuples(index=False):
            message = str(row.error_message).replace("\n", " ").strip()
            lines.append(
                f"| {row.dataset} | {row.scenario} | {row.algorithm} | {row.error_type} | "
                f"{row.failure_count} | {message} |"
            )
        lines.extend(
            [
                "",
                "See `failure_summary.csv` and `logs/failures.json` for the complete failure list with stack traces.",
            ]
        )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path
