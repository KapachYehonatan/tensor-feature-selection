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
    "bh_pairs_bacteria", "bh_bacteria", "bh_genes",
    "clustered_plus_pairs", "bernoulli_correlation_join", "jaccard_join",
]
COLORS = dict(zip(METHOD_ORDER, [
    "#9C6ADE", "#F58518", "#54A24B", "#B279A2", "#4C78A8", "#E45756",
]))
METHOD_LABELS = {
    "bh_pairs": "BH pairs", "bh_pairs_bacteria": "BH pairs + bacteria",
    "bh_bacteria": "BH bacteria",
    "bh_genes": "BH genes", "clustered_marginals": "Clustered marginals",
    "clustered_plus_pairs": "Clustered marginals + pairs",
    "bernoulli_correlation_join": "Bernoulli-correlation-join",
    "jaccard_join": "Jaccard join",
}


def discovery_rates(selected_positive, selected_total, positive_total, universe_total):
    false_positive = selected_total - selected_positive
    null_total = universe_total - positive_total
    return {
        "true_positives": selected_positive,
        "false_positives": false_positive,
        "positives": positive_total,
        "nulls": null_total,
        "selected": selected_total,
        "fdp": false_positive / selected_total if selected_total else 0.0,
        "fpr": false_positive / null_total if null_total else 0.0,
        "power": selected_positive / positive_total if positive_total else np.nan,
    }


def synthetic_discovery_metrics():
    """Score cached BH discoveries against the planted synthetic features."""
    rows = []
    data_root = PROJECT / CONFIGURATION["paths"]["data_root"] / SYNTHETIC_PRESET
    paths = ROOT.glob("synthetic/n_*/seed_*/full_signals/common_clusters/details.npz")
    for details_path in sorted(paths):
        run = details_path.parents[1]
        n_name, seed_name = run.parents[1].name, run.parent.name
        if (n_name, seed_name) not in SYNTHETIC_RUNS:
            continue
        truth_root = data_root / n_name / seed_name / "full_signals"
        signal = pd.read_csv(truth_root / "signal_truth.csv.gz")
        bacteria_truth = pd.read_csv(truth_root / "bacteria_truth.csv.gz")
        gene_truth = pd.read_csv(truth_root / "gene_truth.csv.gz")
        planted_bacteria = set(signal.loc[signal["level"].eq("bacterium"), "cluster"].dropna().astype(int))
        planted_genes = set(signal.loc[signal["level"].eq("gene"), "cluster"].dropna().astype(int))
        q = len(gene_truth)
        planted_pairs = {
            int(row.bacterium) * q + int(row.gene)
            for row in signal[signal["level"].eq("pair")].itertuples()
        }
        with np.load(details_path) as details:
            pair_ids = np.flatnonzero(details["pair_mask"])
            active_bacteria = details["active_bacteria"]
            active_genes = details["active_genes"]
            bacteria_labels = details["bacteria_labels"]
            gene_labels = details["gene_labels"]

        individual_bacteria = {
            int(item): int(bacteria_truth.loc[int(item), "true_cluster"]) in planted_bacteria
            for item in active_bacteria
        }
        individual_genes = {
            int(item): int(gene_truth.loc[int(item), "true_cluster"]) in planted_genes
            for item in active_genes
        }
        clustered_bacteria = {
            int(cluster): any(individual_bacteria[int(item)] for item in active_bacteria[bacteria_labels == cluster])
            for cluster in np.unique(bacteria_labels)
        }
        clustered_genes = {
            int(cluster): any(individual_genes[int(item)] for item in active_genes[gene_labels == cluster])
            for cluster in np.unique(gene_labels)
        }
        pairs = {int(item): int(item) in planted_pairs for item in pair_ids}
        universes = {
            "bh_pairs": {("pair", key): value for key, value in pairs.items()},
            "bh_bacteria": {("bacterium", key): value for key, value in individual_bacteria.items()},
            "bh_genes": {("gene", key): value for key, value in individual_genes.items()},
        }
        universes["bh_pairs_bacteria"] = {
            **universes["bh_pairs"], **universes["bh_bacteria"]
        }
        marginal = {
            **{("bacterium", key): value for key, value in clustered_bacteria.items()},
            **{("gene", key): value for key, value in clustered_genes.items()},
        }
        universes["clustered_marginals"] = marginal
        universes["clustered_plus_pairs"] = {
            **marginal, **{("pair", key): value for key, value in pairs.items()}
        }

        for method, universe in universes.items():
            if method not in METHOD_ORDER:
                continue
            selected_path = run / method / "selected_features.csv.gz"
            marker_path = run / method / "complete.json"
            if not selected_path.exists() or not marker_path.exists():
                continue
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if marker.get("cache_version") != CACHE_VERSION:
                continue
            selected = pd.read_csv(selected_path)
            keys = [(row.level, int(row.feature_id)) for row in selected.itertuples()]
            selected_positive = sum(bool(universe.get(key, False)) for key in keys)
            row = discovery_rates(selected_positive, len(keys), sum(universe.values()), len(universe))
            rows.append({
                "dataset": n_name, "n": int(n_name.removeprefix("n_")),
                "seed": int(seed_name.removeprefix("seed_")), "method": method,
                "q_threshold": float(CONFIGURATION["evaluation"]["screen_fdr"]), **row,
            })
    return pd.DataFrame(rows)


def plot_discovery(metrics):
    if metrics.empty:
        return
    columns = ["fdp", "fpr", "power"]
    means = metrics.groupby(["n", "method"], as_index=False)[columns].mean()
    errors = metrics.groupby(["n", "method"], as_index=False)[columns].std()
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    for axis, metric in zip(axes, columns):
        for method in METHOD_ORDER:
            data = means[means["method"].eq(method)].sort_values("n")
            if data.empty:
                continue
            error = errors[errors["method"].eq(method)].set_index("n").reindex(data["n"])[metric].fillna(0)
            axis.errorbar(data["n"], data[metric], yerr=error, marker="o", color=COLORS[method], label=METHOD_LABELS[method])
        limits = data_limits(metrics[metric]) if metric == "fpr" else (-0.02, 1.02)
        axis.set(title=metric.upper(), xlabel="Samples", ylim=limits)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Proportion")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    figure.suptitle("Synthetic full_signals: BH discoveries (seed mean ± SD)")
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    figure.savefig(FIGURES / "synthetic_discovery.png", dpi=200, bbox_inches="tight")
    plt.close(figure)


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
            for level in ("bacteria", "gene", "pair", "pair_cluster", "total"):
                column = f"{prefix}_{level}_features"
                row[column] = (
                    float(group[column].fillna(0).mean()) if column in group else 0.0
                )
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
    plt.close(figure)


def collect_resources():
    rows = []
    for marker in sorted(ROOT.glob("real/*/*/complete.json")) + sorted(
        ROOT.glob("synthetic/n_*/seed_*/full_signals/*/complete.json")
    ):
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        if metadata.get("cache_version") != CACHE_VERSION:
            continue
        if marker.parent.name not in METHOD_ORDER:
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
    metrics = metrics[metrics["method"].isin(METHOD_ORDER)].copy()
    rsp = rsp[rsp["method"].isin(METHOD_ORDER)].copy()
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(ROOT / "metrics_by_k.csv", index=False, float_format="%.8g")
    rsp.to_csv(ROOT / "rsp.csv", index=False, float_format="%.8g")
    collect_resources().to_csv(ROOT / "resources.csv", index=False, float_format="%.8g")
    discovery = synthetic_discovery_metrics()
    discovery.to_csv(ROOT / "synthetic_discovery_metrics.csv", index=False, float_format="%.8g")
    plot_discovery(discovery)

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
