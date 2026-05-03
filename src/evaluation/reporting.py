from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return markdown_path
