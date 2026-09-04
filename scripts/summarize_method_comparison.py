#!/usr/bin/env python3
"""Build comparison tables and figures from cached per-method results."""

from pathlib import Path
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT / "outputs/method_comparison"
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"
METRICS = ["auroc", "precision", "recall", "f1"]
CACHE_VERSION = 6
SYNTHETIC_PRESET = os.environ.get("SYNTHETIC_PRESET", "main")
CONFIGURATION = json.loads((PROJECT / "synthetic_config.json").read_text(encoding="utf-8"))
PRESET = CONFIGURATION["presets"][SYNTHETIC_PRESET]
SYNTHETIC_RUNS = {
    (f"n_{int(n)}", f"seed_{int(seed)}")
    for n in PRESET["sample_sizes"] for seed in PRESET["seeds"]
}
METHOD_ORDER = [
    "bh_pairs", "bh_bacteria", "bh_genes",
    "clustered_marginals", "clustered_plus_pairs",
]
COLORS = dict(zip(METHOD_ORDER, ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2"]))


def read_cached(filename):
    tables = []
    for path in sorted(ROOT.glob(f"real/*/*/{filename}")):
        marker = json.loads((path.parent / "complete.json").read_text(encoding="utf-8"))
        if marker.get("cache_version") == CACHE_VERSION:
            tables.append(pd.read_csv(path))
    for path in sorted(ROOT.glob(f"synthetic/n_*/seed_*/full_signals/*/{filename}")):
        if (path.parents[3].name, path.parents[2].name) not in SYNTHETIC_RUNS:
            continue
        marker = json.loads((path.parent / "complete.json").read_text(encoding="utf-8"))
        if marker.get("cache_version") == CACHE_VERSION:
            tables.append(pd.read_csv(path))
    return pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()


def seed_average(metrics):
    synthetic = metrics[metrics["source"].eq("synthetic")]
    if synthetic.empty:
        return synthetic
    keys = ["source", "dataset", "n", "scenario", "method", "method_label", "requested_k"]
    numeric = [column for column in synthetic.select_dtypes(include="number") if column not in {"n", "seed", "requested_k"}]
    return synthetic.groupby(keys, as_index=False, dropna=False)[numeric].mean()


def curve_summary(curves, keys):
    rows = []
    for values, group in curves.groupby(keys, dropna=False, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        row = dict(zip(keys, values))
        for metric in METRICS:
            scores = group[metric]
            maximum = scores.max()
            row[f"mean_{metric}"] = float(scores.mean())
            row[f"max_{metric}"] = float(maximum)
            row[f"best_k_{metric}"] = int(group.loc[scores.eq(maximum), "requested_k"].min())
        for prefix in ("output", "selected"):
            for level in ("bacteria", "gene", "pair", "total"):
                column = f"{prefix}_{level}_features"
                row[column] = float(group[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(table):
    shown = table.copy()
    for column in shown.select_dtypes(include="number"):
        shown[column] = shown[column].map(
            lambda value: "" if pd.isna(value) else f"{value:.3f}"
        )
    header = "| " + " | ".join(shown.columns) + " |"
    rule = "| " + " | ".join(["---"] * len(shown.columns)) + " |"
    body = ["| " + " | ".join(map(str, row)) + " |" for row in shown.to_numpy()]
    return "\n".join([header, rule, *body])


def save_table(table, name, title):
    table.to_csv(TABLES / f"{name}.csv", index=False, float_format="%.8g")
    (TABLES / f"{name}.md").write_text(
        f"# {title}\n\n{markdown_table(table)}\n", encoding="utf-8"
    )


def data_limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return 0, 1
    padding = max(0.03, 0.08 * (values.max() - values.min()))
    return max(0, values.min() - padding), min(1, values.max() + padding)


def plot_curves(curves, title, name, errors=None):
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method in METHOD_ORDER:
            data = curves[curves["method"].eq(method)].sort_values("requested_k")
            if data.empty:
                continue
            axis.plot(
                data["requested_k"], data[metric], marker="o", linewidth=2,
                color=COLORS[method], label=data["method_label"].iloc[0],
            )
            if errors is not None:
                error = errors[
                    errors["method"].eq(method)
                ].set_index("requested_k").reindex(data["requested_k"])[metric].fillna(0).to_numpy()
                mean = data[metric].to_numpy()
                axis.fill_between(
                    data["requested_k"], np.maximum(0, mean - error),
                    np.minimum(1, mean + error), color=COLORS[method], alpha=0.15,
                )
        axis.set_title(metric.upper())
        axis.set_xlabel("Top K requested")
        axis.set_ylabel(metric.upper())
        axis.set_ylim(data_limits(curves[metric]))
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=9)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(FIGURES / f"{name}.png", dpi=200, bbox_inches="tight")
    figure.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(figure)


def collect_resources():
    rows = []
    for marker in sorted(ROOT.glob("real/*/*/complete.json")) + sorted(
        ROOT.glob("synthetic/n_*/seed_*/full_signals/*/complete.json")
    ):
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("cache_version") != CACHE_VERSION:
            continue
        parts = marker.parts
        source = "real" if "real" in parts else "synthetic"
        rows.append({
            "source": source,
            "dataset": marker.parents[1].name if source == "real" else marker.parents[3].name,
            "seed": np.nan if source == "real" else int(marker.parents[2].name.removeprefix("seed_")),
            "method": marker.parent.name,
            "runtime_seconds": metadata.get("runtime_seconds", np.nan),
            "peak_rss_mb": metadata.get("peak_rss_mb", np.nan),
        })
    return pd.DataFrame(rows)


def summarize():
    metrics = read_cached("metrics.csv")
    if metrics.empty:
        raise FileNotFoundError(f"No completed comparison metrics under {ROOT}")
    rsp = read_cached("rsp.csv")
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ROOT / "metrics_by_k.csv", index=False, float_format="%.8g")
    rsp.to_csv(ROOT / "rsp.csv", index=False, float_format="%.8g")
    collect_resources().to_csv(ROOT / "resources.csv", index=False, float_format="%.8g")

    real = metrics[metrics["source"].eq("real")].copy()
    synthetic_runs = metrics[metrics["source"].eq("synthetic")].copy()
    synthetic = seed_average(metrics)
    summaries = []
    if not real.empty:
        summaries.append(curve_summary(real, ["source", "dataset", "method", "method_label"]))
    if not synthetic.empty:
        summaries.append(curve_summary(synthetic, ["source", "dataset", "n", "method", "method_label"]))
    dataset_summary = pd.concat(summaries, ignore_index=True)

    if not rsp.empty:
        real_rsp = rsp[rsp["source"].eq("real")].groupby(
            ["source", "dataset", "method"], as_index=False
        )["rsp"].mean()
        synthetic_rsp = rsp[rsp["source"].eq("synthetic")].groupby(
            ["source", "dataset", "method"], as_index=False
        )["rsp"].mean()
        rsp_summary = pd.concat([real_rsp, synthetic_rsp], ignore_index=True)
        dataset_summary = dataset_summary.merge(
            rsp_summary, on=["source", "dataset", "method"], how="left"
        )
    else:
        dataset_summary["rsp"] = np.nan
    dataset_summary.to_csv(ROOT / "dataset_summary.csv", index=False, float_format="%.8g")

    summary_values = [
        column for column in dataset_summary.select_dtypes(include="number")
        if column.startswith(("mean_", "max_", "output_", "selected_")) or column == "rsp"
    ]
    method_summary = dataset_summary.groupby(
        ["source", "method", "method_label"], as_index=False, dropna=False
    )[summary_values].mean()
    method_summary.to_csv(ROOT / "method_summary.csv", index=False, float_format="%.8g")
    save_table(method_summary, "methods", "Average method results")

    for dataset, group in real.groupby("dataset"):
        table = dataset_summary[
            dataset_summary["source"].eq("real") & dataset_summary["dataset"].eq(dataset)
        ]
        save_table(table, f"real_{dataset}", f"Real dataset: {dataset}")
        plot_curves(group, f"Real dataset: {dataset}", f"real_{dataset}")

    for n, group in synthetic_runs.groupby("n"):
        means = group.groupby(
            ["method", "method_label", "requested_k"], as_index=False
        )[METRICS].mean()
        errors = group.groupby(
            ["method", "method_label", "requested_k"], as_index=False
        )[METRICS].std()
        table = dataset_summary[
            dataset_summary["source"].eq("synthetic") & dataset_summary["n"].eq(n)
        ]
        save_table(table, f"synthetic_n_{int(n)}", f"Synthetic full_signals: n={int(n)}")
        plot_curves(
            means, f"Synthetic full_signals: n={int(n)} (seed mean ± SD)",
            f"synthetic_n_{int(n)}", errors,
        )

    if not real.empty:
        real_mean = real.groupby(
            ["method", "method_label", "requested_k"], as_index=False
        )[METRICS].mean()
        real_std = real.groupby(
            ["method", "method_label", "requested_k"], as_index=False
        )[METRICS].std()
        plot_curves(real_mean, "Real datasets: equal-weight mean ± SD", "aggregate_real", real_std)
    if not synthetic.empty:
        synthetic_mean = synthetic.groupby(
            ["method", "method_label", "requested_k"], as_index=False
        )[METRICS].mean()
        synthetic_std = synthetic.groupby(
            ["method", "method_label", "requested_k"], as_index=False
        )[METRICS].std()
        plot_curves(
            synthetic_mean, "Synthetic full_signals: equal-weight n mean ± SD",
            "aggregate_synthetic", synthetic_std,
        )

    print(
        f"Summarized {metrics[['source', 'dataset', 'seed']].drop_duplicates().shape[0]} "
        f"runs in {ROOT}", flush=True,
    )


if __name__ == "__main__":
    summarize()
