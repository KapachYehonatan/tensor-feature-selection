#!/usr/bin/env python3
"""Extend the original QinJ fine-K plot using completed, compatible caches."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.summarize_method_comparison import data_limits, METRICS

COMPARISON = PROJECT / "outputs/method_comparison"
SIDE = PROJECT / "outputs/qinj_redundancy_analysis"
NAME = "fine_k_real_QinJ_2012_updated"
K_VALUES = tuple(range(1, 11))
OLD_METHODS = {
    "bh_pairs_bacteria": "Raw pairs + bacterial means | BH",
    "bh_bacteria": "Bacterial means | BH",
    "bh_genes": "Gene means | BH",
    "clustered_plus_pairs": "Marginal co-clusters + pairs | BH",
    "bernoulli_correlation_join": "Bernoulli-join mean | BH",
}
NEW_METHODS = {
    "penalty": "Bernoulli-join mean | MW effect minus max-|r| penalty",
    "penalty_mean": "Bernoulli-join mean | MW effect minus mean-|r| penalty",
    "penalty_r2": "Bernoulli-join mean | MW effect minus R² penalty",
    "penalty_residual_target": "Bernoulli-join mean | MW effect minus residual penalty",
    "merge": "Bernoulli-join mean + additional correlation merging | BH",
}
COLORS = dict(zip([*OLD_METHODS, *NEW_METHODS], [
    "#9C6ADE", "#F58518", "#54A24B", "#B279A2", "#4C78A8",
    "#E45756", "#17BECF", "#8C564B", "#D45087", "#237A57",
]))


def load_comparison(original, side, dataset="QinJ_2012", k_values=K_VALUES):
    original = original[original.source.eq("real") & original.dataset.eq(dataset)
                        & original.requested_k.isin(k_values)].copy()
    side = side[side.requested_k.isin(k_values)].copy()
    for table, methods in ((original, OLD_METHODS), (side, {**NEW_METHODS, "baseline": "", "bacteria": ""})):
        for method in methods:
            curve = table[table.method.eq(method)]
            if len(curve) != len(k_values) or set(curve.requested_k) != set(k_values):
                raise ValueError(f"Incomplete or duplicate K results: {dataset}/{method}")
            if not np.isfinite(curve[list(METRICS)]).all().all():
                raise ValueError(f"Nonfinite metrics: {method}")
    for old, new in (("bernoulli_correlation_join", "baseline"), ("bh_bacteria", "bacteria")):
        a = original[original.method.eq(old)].sort_values("requested_k")
        b = side[side.method.eq(new)].sort_values("requested_k")
        if not np.allclose(a[list(METRICS)], b[list(METRICS)], rtol=0, atol=1e-7):
            raise ValueError(f"Original and side-analysis baselines disagree: {old}")
    original = original[original.method.isin(OLD_METHODS)].copy()
    original["method_label"] = original.method.map(OLD_METHODS)
    side = side[side.method.isin(NEW_METHODS)].copy()
    side["method_label"] = side.method.map(NEW_METHODS)
    side["source"], side["dataset"] = "real", dataset
    original["result_origin"] = "original method comparison"
    side["result_origin"] = "training-CV redundancy analysis"
    return pd.concat([original, side], ignore_index=True)


def plot_comparison(data, title, path, k_values=K_VALUES, errors=None, extra_methods=None):
    extra_methods = extra_methods or {}
    labels = {**OLD_METHODS, **NEW_METHODS, **extra_methods}
    colors = {**COLORS, **dict(zip(extra_methods, ("#111111", "#BCBD22")))}
    for method in NEW_METHODS:
        parameter = data.loc[data.method.eq(method), "parameter"].unique()
        if len(parameter) != 1 and errors is None:
            raise ValueError(f"Expected one locked setting: {method}")
        if errors is None:
            labels[method] += f" ({'r' if method == 'merge' else 'λ'}={parameter[0]:.2f})"
    figure, axes = plt.subplots(2, 2, figsize=(16, 11), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        for method, label in labels.items():
            curve = data[data.method.eq(method)].sort_values("requested_k")
            axis.plot(curve.requested_k, curve[metric], color=colors[method],
                      marker="D" if method == "merge" else "^" if method in NEW_METHODS else "o",
                      linestyle="--" if method in NEW_METHODS else "-",
                      linewidth=3 if method in extra_methods else 1.8,
                      markersize=5, label=label)
            if method in extra_methods and metric == "auroc" and errors is None:
                chosen = curve[curve.requested_k.eq(curve.cv_selected_k)]
                axis.scatter(chosen.requested_k, chosen[metric], marker="*", s=180,
                             color=colors[method], edgecolor="white", zorder=5)
            if errors is not None:
                spread = errors[errors.method.eq(method)].set_index("requested_k").reindex(curve.requested_k)[metric].fillna(0).to_numpy()
                axis.fill_between(curve.requested_k, np.clip(curve[metric]-spread, 0, 1),
                                  np.clip(curve[metric]+spread, 0, 1), color=colors[method], alpha=.08)
        axis.set(title=metric.upper(), xlabel="Top K requested", ylabel=metric.upper(),
                 xticks=k_values, ylim=data_limits(data[metric]))
        axis.grid(alpha=.25)
    handles, names = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles[:5], names[:5], loc="lower left", bbox_to_anchor=(.02, .045),
                  fontsize=9, title="Original methods: representation | selection")
    figure.legend(handles[5:], names[5:], loc="lower left", bbox_to_anchor=(.51, .045),
                  fontsize=9, title="New join-only methods: representation | selection")
    figure.suptitle(title, fontsize=16)
    figure.tight_layout(rect=(0, .25 if extra_methods else .23, 1, .93 if extra_methods else .97))
    figure.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main():
    marker = SIDE / "complete.json"
    if not marker.exists():
        raise ValueError("Complete the side analysis before plotting")
    complete = json.loads(marker.read_text())
    if complete.get("status") != "complete" or complete["signature"] != json.loads((SIDE / "protocol.json").read_text())["signature"]:
        raise ValueError("Side-analysis completion and protocol signatures disagree")
    data = load_comparison(pd.read_csv(COMPARISON / "fine_k_real_metrics.csv"),
                           pd.read_csv(SIDE / "metrics.csv"))
    path = COMPARISON / "figures" / f"{NAME}.png"
    plot_comparison(data, "QinJ_2012 — updated comparison (K=1,…,10)", path)
    for method in NEW_METHODS:
        parameter = data.loc[data.method.eq(method), "parameter"].iloc[0]
        data.loc[data.method.eq(method), "method_label"] += f" ({'r' if method == 'merge' else 'λ'}={parameter:.2f})"
    data.to_csv(COMPARISON / "tables" / f"{NAME}.csv", index=False)
    print(f"Saved {len(OLD_METHODS)+len(NEW_METHODS)} methods × {len(K_VALUES)} K values: {path}")


if __name__ == "__main__":
    main()
