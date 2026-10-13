#!/usr/bin/env python3
"""Plot the requested BH and Bernoulli-join methods from cached metrics."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
COMPARISON = PROJECT / "outputs/method_comparison"
VARIATIONS = PROJECT / "outputs/bernoulli_variations/tables/metrics_by_k.csv"
OUTPUT = COMPARISON / "figures/selected_methods"
METRICS = ("auroc", "precision", "recall", "f1")
COARSE_K = (5, 10, 20, 35, 50, 60, 75, 85, 100)
FINE_K = tuple(range(1, 11))
METHODS = {
    "bh_pairs_bacteria": "BH Top-K pairs + bacteria-collapsed",
    "bh_bacteria": "BH Top-K bacteria-collapsed",
    "bh_genes": "BH Top-K gene-collapsed",
    "baseline": "Bernoulli-correlation-join · mean",
    "max_variance": "Bernoulli-correlation-join · max variance",
    "coprojection_positive": "2D Bernoulli-correlation clustering · positive",
    "coprojection_absolute": "2D Bernoulli-correlation clustering · abs",
}
COLORS = dict(zip(METHODS, (
    "#9C6ADE", "#F58518", "#54A24B", "#4C78A8", "#E45756",
    "#72B7B2", "#B279A2",
)))


def load_metrics(main_path, k_values):
    main = pd.read_csv(main_path)
    variation = pd.read_csv(VARIATIONS)
    main = main[main["source"].eq("real") & main["method"].isin(METHODS)]
    variation = variation[
        variation["source"].eq("real") & variation["method"].isin(METHODS)
    ]
    data = pd.concat((main, variation), ignore_index=True)
    data = data[data["requested_k"].isin(k_values)].copy()
    data["method_label"] = data["method"].map(METHODS)
    data = data.drop_duplicates(["dataset", "method", "requested_k"], keep="last")
    expected = set(METHODS)
    missing = {
        dataset: sorted(expected - set(group["method"]))
        for dataset, group in data.groupby("dataset")
        if set(group["method"]) != expected
    }
    if missing:
        raise ValueError(f"Incomplete cached methods: {missing}")
    return data


def limits(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    spread = values.max() - values.min()
    padding = max(0.03, 0.08 * spread)
    return max(0, values.min() - padding), min(1, values.max() + padding)


def plot(data, title, path, errors=None):
    figure, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method, label in METHODS.items():
            curve = data[data["method"].eq(method)].sort_values("requested_k")
            axis.plot(
                curve["requested_k"], curve[metric], marker="o", linewidth=2,
                color=COLORS[method], label=label,
            )
            if errors is not None:
                error = errors[errors["method"].eq(method)].set_index(
                    "requested_k"
                ).reindex(curve["requested_k"])[metric].fillna(0).to_numpy()
                mean = curve[metric].to_numpy()
                axis.fill_between(
                    curve["requested_k"], np.maximum(0, mean - error),
                    np.minimum(1, mean + error), color=COLORS[method], alpha=0.14,
                )
        axis.set_title(metric.upper())
        axis.set_xlabel("Top K requested")
        axis.set_ylabel(metric.upper())
        axis.set_ylim(limits(data[metric]))
        axis.grid(alpha=0.25)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle(title)
    figure.tight_layout(rect=(0, 0.13, 1, 1))
    figure.savefig(path, dpi=250, bbox_inches="tight")
    plt.close(figure)


def render(name, main_path, k_values):
    data = load_metrics(main_path, k_values)
    data.to_csv(OUTPUT / f"metrics_{name}.csv", index=False, float_format="%.8g")
    for dataset, group in data.groupby("dataset", sort=True):
        plot(group, f"Real dataset: {dataset}", OUTPUT / f"{name}_{dataset}.png")
    keys = ["method", "method_label", "requested_k"]
    mean = data.groupby(keys, as_index=False)[list(METRICS)].mean()
    error = data.groupby(keys, as_index=False)[list(METRICS)].std()
    plot(
        mean, "Real datasets: equal-weight mean ± SD",
        OUTPUT / f"{name}_aggregate_real.png", error,
    )
    return data["dataset"].nunique()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    coarse = render("coarse", COMPARISON / "metrics_by_k.csv", COARSE_K)
    fine = render("fine_k", COMPARISON / "fine_k_real_metrics.csv", FINE_K)
    print(f"Saved combined figures for {coarse} coarse-grid and {fine} fine-grid datasets to {OUTPUT}")


if __name__ == "__main__":
    main()
