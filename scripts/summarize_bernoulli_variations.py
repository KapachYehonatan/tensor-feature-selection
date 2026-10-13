#!/usr/bin/env python3
"""Rebuild Bernoulli-variation tables and PNG figures from completed caches only."""

from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_bernoulli_variations import (  # noqa: E402
    ALL_K, CLUSTER_VERSION, COPROJECTIONS, COPROJECTION_VERSION,
    K_VALUES, LABELS, OUTPUT_ROOT, ROOT, VARIATIONS,
)
from scripts.run_method_comparison import real_jobs  # noqa: E402

METRICS = ("auroc", "precision", "recall", "f1")
COLORS = dict(zip(LABELS, (
    "#4C78A8", "#F58518", "#54A24B", "#B279A2", "#E45756",
    "#72B7B2", "#FF9DA6",
)))


def markdown_table(frame):
    """Small Markdown table without pandas' optional tabulate dependency."""
    columns = list(frame.columns)
    lines = ["| " + " | ".join(columns) + " |",
             "| " + " | ".join("---" for _ in columns) + " |"]
    for row in frame.itertuples(index=False, name=None):
        cells = [f"{value:.3f}" if isinstance(value, (float, np.floating)) else str(value)
                 for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def cached_metrics():
    rows, rsp, status = [], [], []
    fine_path = OUTPUT_ROOT / "fine_k_real_metrics.csv"
    fine = pd.read_csv(fine_path) if fine_path.exists() else pd.DataFrame()
    for job in real_jobs():
        dataset = job["dataset"]
        baseline = OUTPUT_ROOT / "real" / dataset / "bernoulli_correlation_join"
        if (baseline / "complete.json").exists():
            coarse = pd.read_csv(baseline / "metrics.csv")
            rows.append(coarse.assign(method="baseline", method_label=LABELS["baseline"]))
            info = json.loads((baseline / "complete.json").read_text())
            status.append({"dataset": dataset, "method": "baseline", "status": "complete",
                           "clusters": info.get("clusters"),
                           "candidate_features": info.get("clusters"),
                           "selected_features": len(pd.read_csv(baseline / "selected_features.csv.gz")),
                           "runtime_seconds": info.get("runtime_seconds"),
                           "peak_rss_mb": info.get("peak_rss_mb")})
            if not fine.empty:
                subset = fine[(fine.dataset == dataset) & (fine.method == "bernoulli_correlation_join")]
                if not subset.empty:
                    rows.append(subset.assign(method="baseline", method_label=LABELS["baseline"]))
            if (baseline / "rsp.csv").exists():
                rsp.append(pd.read_csv(baseline / "rsp.csv").assign(method="baseline"))
        for variation in VARIATIONS:
            path = ROOT / dataset / variation
            marker = path / "complete.json"
            if variation in {"signed_families", "kaminpar"}:
                cluster_marker = path / "community/variation.json"
                if not cluster_marker.exists() or json.loads(cluster_marker.read_text()).get("version") != CLUSTER_VERSION:
                    if marker.exists():
                        status.append({"dataset": dataset, "method": variation,
                                       "status": "stale_clustering_cache"})
                    continue
            if marker.exists():
                info = json.loads(marker.read_text())
                if (variation in COPROJECTIONS
                        and info.get("architecture_version") != COPROJECTION_VERSION):
                    status.append({"dataset": dataset, "method": variation,
                                   "status": "stale_architecture"})
                    continue
                rows.append(pd.read_csv(path / "metrics.csv"))
                rsp.append(pd.read_csv(path / "rsp.csv"))
                status.append({"dataset": dataset, "method": variation, "status": "complete",
                               "clusters": info.get("clusters"),
                               "candidate_features": info.get("candidate_features"),
                               "selected_features": info.get("selected_features"),
                               "selected_positive_mean": info.get("selected_positive_mean"),
                               "selected_negative_contrast": info.get("selected_negative_contrast"),
                               "bacteria_clusters": info.get("bacteria_clusters"),
                               "gene_clusters": info.get("gene_clusters"),
                               "selected_bacteria": info.get("selected_bacteria"),
                               "selected_genes": info.get("selected_genes"),
                               "rounds": info.get("rounds"),
                               "converged": info.get("converged"),
                               "clustering_runtime_seconds": info.get("clustering_runtime_seconds"),
                               "runtime_seconds": info.get("runtime_seconds"),
                               "peak_rss_mb": info.get("peak_rss_mb")})
            elif (path / "resource_limit.json").exists():
                limit = json.loads((path / "resource_limit.json").read_text())
                status.append({"dataset": dataset, "method": variation,
                               "status": limit.get("status", "failed"),
                               "reason": limit.get("reason", "")[-400:]})
    if not rows:
        raise FileNotFoundError("No completed baseline or variation caches were found")
    metrics = pd.concat(rows, ignore_index=True).drop_duplicates(
        ["dataset", "method", "requested_k"], keep="first")
    return metrics, pd.concat(rsp, ignore_index=True), pd.DataFrame(status)


def summarize_rows(metrics):
    summaries = []
    for (dataset, method), group in metrics.groupby(["dataset", "method"], sort=False):
        group = (group[group.requested_k.isin(K_VALUES)]
                 .drop_duplicates("requested_k", keep="first")
                 .sort_values("requested_k"))
        if group.empty:
            continue
        row = {"dataset": dataset, "method": method, "method_label": LABELS[method],
               "coarse_k_observed": len(group),
               "mean_effective_k": group.effective_k.mean()}
        for metric in METRICS:
            row[f"mean_{metric}"] = group[metric].mean()
            maximum = group[metric].max()
            row[f"max_{metric}"] = maximum
            row[f"best_k_{metric}"] = int(group.loc[group[metric] == maximum, "requested_k"].min())
        summaries.append(row)
    dataset = pd.DataFrame(summaries)
    baseline = dataset[dataset.method == "baseline"].set_index("dataset")["mean_auroc"]
    dataset["delta_mean_auroc_vs_baseline"] = [
        row.mean_auroc - baseline.get(row.dataset, np.nan) for row in dataset.itertuples()]
    numeric = [column for column in dataset if column.startswith(("mean_", "max_", "delta_"))]
    aggregate = dataset.groupby("method", as_index=False)[numeric].mean()
    aggregate.insert(1, "method_label", aggregate.method.map(LABELS))
    aggregate["datasets"] = dataset.groupby("method").size().reindex(aggregate.method).to_numpy()
    return dataset, aggregate


def plot_curves(data, title, path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for method, group in data.groupby("method"):
            curve = group.groupby("requested_k", as_index=False)[metric].mean()
            ax.plot(curve.requested_k, curve[metric], marker="o", markersize=3,
                    label=LABELS[method], color=COLORS[method])
        ax.set_title(metric.upper() if metric == "auroc" else metric.capitalize())
        ax.set_ylabel(metric.capitalize())
        ax.grid(alpha=0.2)
    for ax in axes[-1]:
        ax.set_xlabel("Top K selected features")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(3, len(labels)))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def summarize():
    metrics, rsp, status = cached_metrics()
    tables, figures = ROOT / "tables", ROOT / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    metrics.sort_values(["dataset", "method", "requested_k"]).to_csv(
        tables / "metrics_by_k.csv", index=False)
    if not rsp.empty:
        rsp.to_csv(tables / "rsp.csv", index=False)
    if not status.empty:
        status.to_csv(tables / "run_status.csv", index=False)
    dataset, _ = summarize_rows(metrics)
    if not status.empty:
        dataset = dataset.merge(status[status.status == "complete"].drop(columns="status"),
                                on=["dataset", "method"], how="left", validate="one_to_one")
    if not rsp.empty:
        dataset = dataset.merge(rsp[["dataset", "method", "rp", "sp", "rsp"]],
                                on=["dataset", "method"], how="left", validate="one_to_one")
    numeric = dataset.select_dtypes(include="number").columns.drop(
        [column for column in ("best_k_auroc", "best_k_precision", "best_k_recall", "best_k_f1")
         if column in dataset.columns])
    aggregate = dataset.groupby("method", as_index=False)[list(numeric)].mean()
    aggregate.insert(1, "method_label", aggregate.method.map(LABELS))
    aggregate["datasets"] = dataset.groupby("method").size().reindex(aggregate.method).to_numpy()
    dataset.to_csv(tables / "dataset_summary.csv", index=False)
    aggregate.to_csv(tables / "method_summary.csv", index=False)
    paired = dataset.pivot(index="dataset", columns="method", values="delta_mean_auroc_vs_baseline")
    paired.to_csv(tables / "paired_auroc_differences.csv")
    for name, group in metrics.groupby("dataset"):
        for grid, suffix in ((K_VALUES, "coarse"), (range(1, 11), "fine")):
            subset = group[group.requested_k.isin(grid)]
            if not subset.empty:
                plot_curves(subset, f"{name} · {suffix} K", figures / f"{name}_{suffix}.png")
        dataset[dataset.dataset == name].to_csv(tables / f"{name}.csv", index=False)
    for grid, suffix in ((K_VALUES, "coarse"), (range(1, 11), "fine")):
        complete_methods = dataset.groupby("method").dataset.nunique()
        complete_methods = complete_methods[complete_methods == len(real_jobs())].index
        subset = metrics[metrics.requested_k.isin(grid) & metrics.method.isin(complete_methods)]
        if not subset.empty:
            plot_curves(subset, f"Equal-weight real-dataset mean · {suffix} K",
                        figures / f"aggregate_{suffix}.png")
    complete = dataset.groupby("method").dataset.nunique()
    eligible = complete[complete == len(real_jobs())].index.tolist()
    eligible = [item for item in eligible if item != "baseline"]
    if eligible:
        winner = aggregate[aggregate.method.isin(eligible)].sort_values("mean_auroc", ascending=False).iloc[0]
        baseline_score = aggregate.loc[aggregate.method == "baseline", "mean_auroc"].iloc[0]
        difference = winner.mean_auroc - baseline_score
        conclusion = (f"Leading completed variation: {LABELS[winner.method]} "
                      f"(mean AUROC {winner.mean_auroc:.3f}; baseline {baseline_score:.3f}; "
                      f"paired difference {difference:+.3f}).")
        if difference < 0:
            conclusion += " Baseline is better on the eight-dataset mean."
        elif difference < 0.005:
            conclusion += " Gain <0.005: prefer the simpler baseline."
    else:
        conclusion = "No variation has completed all eight datasets; no winner is declared."
    report = ["# Bernoulli-join variation study", "", conclusion, "",
              "This is exploratory: the same holdout selects the variation, so final performance needs fresh splits.",
              "Reported peak RSS is the largest recorded coordinator-process peak; parallel Leiden worker RSS is not summed.",
              "Baseline runtime includes its original join, whereas variation runtimes reuse that completed graph.",
              "", "## Completed method means", "", markdown_table(aggregate),
              "", "## Paired mean-AUROC differences from baseline", "",
              markdown_table(paired.reset_index()), ""]
    (ROOT / "REPORT.md").write_text("\n".join(report), encoding="utf-8")
    print(conclusion)
    print(f"Cache-only report: {ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    summarize()
