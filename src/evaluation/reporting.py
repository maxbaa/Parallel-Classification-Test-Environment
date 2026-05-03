from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SUMMARY_COLUMNS = [
    "accuracy",
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


def build_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty:
        return pd.DataFrame()

    grouped = (
        results_df.assign(success_flag=results_df["status"].eq("success").astype(int))
        .groupby(["dataset", "scenario", "algorithm"], dropna=False)
        .agg(
            runs_total=("run_index", "count"),
            runs_success=("success_flag", "sum"),
            accuracy_mean=("accuracy", "mean"),
            accuracy_std=("accuracy", "std"),
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
    return grouped.sort_values(["dataset", "scenario", "algorithm"]).reset_index(drop=True)


def _plot_metric(summary_df: pd.DataFrame, metric: str, ylabel: str, output_path: Path) -> Path | None:
    if summary_df.empty or metric not in summary_df.columns:
        return None

    plot_df = summary_df.copy()
    plot_df["context"] = plot_df["dataset"] + " | " + plot_df["scenario"]
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
    plot_df["context"] = plot_df["dataset"] + " | " + plot_df["scenario"]
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
    plot_df = plot_df.sort_values(["dataset", "scenario", "algorithm"]).reset_index(drop=True)
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
    plot_df["context"] = plot_df["dataset"] + " | " + plot_df["scenario"]
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

    if "gpu_max_gb_mean" in summary_df.columns and not summary_df["gpu_max_gb_mean"].fillna(0).eq(0).all():
        plot_paths.append(
            _plot_metric(summary_df, "gpu_max_gb_mean", "GPU Memory Peak (GB)", output_dir / "gpu_memory.png")
        )
    else:
        plot_paths.append(
            _plot_metric(summary_df, "ram_max_gb_mean", "RAM Peak (GB)", output_dir / "ram_peak.png")
        )

    return [path for path in plot_paths if path is not None]


def write_summary_markdown(summary_df: pd.DataFrame, output_dir: Path) -> Path:
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
        lines.extend(
            [
                "## Top Accuracy",
                "",
                "| Dataset | Scenario | Algorithm | Accuracy | F1 | CV Score | Train Time (s) | Success Rate |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in best_accuracy.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.scenario} | {row.algorithm} | "
                f"{row.accuracy_mean:.4f} | {row.f1_mean:.4f} | {row.best_cv_score_mean:.4f} | {row.train_time_sec_mean:.4f} | "
                f"{row.success_rate:.2%} |"
            )
        lines.extend(
            [
                "",
                "## Reliability Overview",
                "",
                "| Dataset | Scenario | Algorithm | Success Rate | Accuracy | Train Time (s) |",
                "| --- | --- | --- | ---: | ---: | ---: |",
            ]
        )
        for row in highest_success.itertuples(index=False):
            lines.append(
                f"| {row.dataset} | {row.scenario} | {row.algorithm} | "
                f"{row.success_rate:.2%} | {row.accuracy_mean:.4f} | {row.train_time_sec_mean:.4f} |"
            )

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path
