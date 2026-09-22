#!/usr/bin/env python3
"""Export figures summarizing the repeated synthetic experiment."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from pandas.errors import EmptyDataError  # noqa: E402


PROJECT = Path(__file__).resolve().parents[1]
PRESET = "main"
RESULTS = PROJECT / "outputs/synthetic_evaluation" / PRESET
FIGURES = RESULTS / "figures"
DPI = 200

SCENARIOS = [
    "exact_blocks",
    "noisy_blocks",
    "null_target",
    "marginal_signals",
    "pair_signals",
    "full_signals",
]
SCENARIO_LABELS = {
    "exact_blocks": "Exact blocks",
    "noisy_blocks": "Noisy blocks",
    "null_target": "Null target",
    "marginal_signals": "Marginal signals",
    "pair_signals": "Pair signals",
    "full_signals": "Marginal + pair signals",
}
MODEL_ORDER = [
    "intercept",
    "bacteria_clusters",
    "gene_clusters",
    "marginal_clusters",
    "pairs_only",
    "marginal_plus_pairs",
]
MODEL_LABELS = {
    "intercept": "Intercept",
    "bacteria_clusters": "Bacteria",
    "gene_clusters": "Genes",
    "marginal_clusters": "Both marginals",
    "pairs_only": "Pairs only",
    "marginal_plus_pairs": "Marginals + pairs",
}
COLORS = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#B279A2", "#72B7B2"]


def load_runs(filename):
    """Read every nonempty per-run table and add its run identifiers."""
    tables = []
    for path in sorted(RESULTS.glob(f"n_*/seed_*/*/{filename}")):
        try:
            table = pd.read_csv(path)
        except EmptyDataError:
            continue
        if table.empty:
            continue
        table.insert(0, "scenario", path.parent.name)
        table.insert(0, "seed", int(path.parents[1].name.removeprefix("seed_")))
        table.insert(0, "n", int(path.parents[2].name.removeprefix("n_")))
        tables.append(table)
    if not tables:
        raise FileNotFoundError(f"No nonempty {filename} files under {RESULTS}")
    return pd.concat(tables, ignore_index=True)


def plot_seed_means(
    ax, data, value, specifications, bounded=False, nonnegative=False
):
    """Plot seed observations and their mean ± one standard deviation."""
    for key, label, color, linestyle in specifications:
        observed = data.loc[data["series"].eq(key), ["n", value]].dropna()
        if observed.empty:
            continue
        ax.scatter(observed["n"], observed[value], color=color, alpha=0.25, s=18)
        scores = observed.groupby("n")[value].agg(["mean", "std"]).reset_index()
        error = scores["std"].fillna(0).to_numpy()
        if bounded:
            mean = scores["mean"].to_numpy()
            error = np.vstack([np.minimum(error, mean), np.minimum(error, 1 - mean)])
        elif nonnegative:
            mean = scores["mean"].to_numpy()
            error = np.vstack([np.minimum(error, mean), error])
        ax.errorbar(
            scores["n"],
            scores["mean"],
            yerr=error,
            label=label,
            color=color,
            linestyle=linestyle,
            marker="o",
            linewidth=2,
            capsize=3,
        )
    ax.set_xticks(sorted(data["n"].unique()))
    ax.set_xlabel("Samples (n)")
    if bounded:
        ax.set_ylim(-0.03, 1.03)
    elif nonnegative:
        ax.set_ylim(bottom=0)


def save_figure(figure, name):
    figure.savefig(FIGURES / f"{name}.png", dpi=DPI, bbox_inches="tight")
    plt.close(figure)


def save_average(table, groups, values, name):
    summary = table.groupby(groups, dropna=False)[values].agg(["mean", "std", "count"])
    summary.columns = ["_".join(column) for column in summary.columns]
    summary.reset_index().to_csv(FIGURES / name, index=False, float_format="%.6f")


def clustering_figure(clustering, truth_counts):
    data = clustering[
        clustering["scope"].eq("pipeline")
        & clustering["scenario"].isin(["exact_blocks", "noisy_blocks"])
    ].copy()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    scenario_colors = {"exact_blocks": COLORS[0], "noisy_blocks": COLORS[1]}

    for row, level in enumerate(["bacterium", "gene"]):
        level_data = data[data["level"].eq(level)].copy()
        agreement = level_data.melt(
            id_vars=["n", "seed", "scenario"],
            value_vars=["ari", "nmi"],
            var_name="agreement",
            value_name="score",
        )
        agreement["series"] = agreement["scenario"] + "_" + agreement["agreement"]
        specs = [
            (
                f"{scenario}_{metric}",
                f"{SCENARIO_LABELS[scenario]} {metric.upper()}",
                scenario_colors[scenario],
                "-" if metric == "ari" else "--",
            )
            for scenario in ["exact_blocks", "noisy_blocks"]
            for metric in ["ari", "nmi"]
        ]
        plot_seed_means(axes[row, 0], agreement, "score", specs, bounded=True)
        axes[row, 0].set_title(
            f"Clustering level: {level.title()}\nMetric: agreement with true clusters"
        )
        axes[row, 0].set_ylabel("ARI / NMI")

        level_data["series"] = level_data["scenario"]
        count_specs = [
            (scenario, SCENARIO_LABELS[scenario], scenario_colors[scenario], "-")
            for scenario in ["exact_blocks", "noisy_blocks"]
        ]
        plot_seed_means(axes[row, 1], level_data, "recovered_clusters", count_specs)
        axes[row, 1].axhline(
            truth_counts[level], color="black", linestyle=":", label="True count"
        )
        axes[row, 1].set_title(
            f"Clustering level: {level.title()}\nMetric: recovered cluster count"
        )
        axes[row, 1].set_ylabel("Clusters")

    axes[0, 0].legend(title="Data scenario / metric", fontsize=8, ncol=2)
    axes[0, 1].legend(title="Data scenario", fontsize=8)
    figure.suptitle("Analysis: co-clustering recovery across independent seeds")
    figure.tight_layout()
    save_figure(figure, "01_coclustering_recovery")


def marginal_selection_figure(selection):
    data = selection[
        selection["level"].isin(["bacterium", "gene"])
        & selection["scenario"].isin(["marginal_signals", "full_signals"])
    ].copy()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    metric_specs = [
        ("precision", "Precision", COLORS[0], "-"),
        ("recall", "Recall", COLORS[1], "-"),
        ("f1", "F1", COLORS[2], "--"),
    ]
    for row, level in enumerate(["bacterium", "gene"]):
        for column, scenario in enumerate(["marginal_signals", "full_signals"]):
            panel = data[
                data["level"].eq(level) & data["scenario"].eq(scenario)
            ].melt(
                id_vars=["n", "seed"],
                value_vars=["precision", "recall", "f1"],
                var_name="series",
                value_name="score",
            )
            plot_seed_means(axes[row, column], panel, "score", metric_specs, bounded=True)
            axes[row, column].set_title(
                f"Selection level: {level.title()}\n"
                f"Data scenario: {SCENARIO_LABELS[scenario]}"
            )
            axes[row, column].set_ylabel("Selection score")
    axes[0, 0].legend(title="Selection metric", fontsize=8)
    figure.suptitle("Analysis: marginal feature selection")
    figure.tight_layout()
    save_figure(figure, "02_marginal_selection")


def pair_selection_figure(selection):
    pairs = selection[selection["level"].eq("pair")].copy()
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    quality_specs = [
        ("precision", "Precision", COLORS[0], "-"),
        ("recall", "Recall", COLORS[1], "-"),
        ("f1", "F1", COLORS[2], "--"),
    ]
    for axis, scenario in zip(axes[0], ["pair_signals", "full_signals"]):
        quality = pairs[pairs["scenario"].eq(scenario)].melt(
            id_vars=["n", "seed"],
            value_vars=["precision", "recall", "f1"],
            var_name="series",
            value_name="score",
        )
        plot_seed_means(axis, quality, "score", quality_specs, bounded=True)
        axis.set_title(
            f"Selection level: Pairs\nData scenario: {SCENARIO_LABELS[scenario]}"
        )
        axis.set_ylabel("Selection score")
        axis.legend(title="Selection metric", fontsize=8)

    false_discovery = pairs[pairs["scenario"].isin([
        "null_target", "marginal_signals", "pair_signals", "full_signals"
    ])].copy()
    false_discovery["series"] = false_discovery["scenario"]
    false_specs = [
        (scenario, SCENARIO_LABELS[scenario], COLORS[index], "-")
        for index, scenario in enumerate(
            ["null_target", "marginal_signals", "pair_signals", "full_signals"]
        )
    ]
    plot_seed_means(
        axes[1, 0], false_discovery, "false_discovery_proportion", false_specs,
        bounded=True,
    )
    axes[1, 0].set_title(
        "Selection level: Pairs\nMetric: false discovery proportion (FP / selected)"
    )
    axes[1, 0].set_ylabel("False discovery proportion")
    axes[1, 0].legend(title="Data scenario", fontsize=8)

    signal_pairs = pairs[pairs["scenario"].isin([
        "pair_signals", "full_signals"
    ])].copy()
    signal_pairs["series"] = signal_pairs["scenario"]
    retention_specs = [
        (scenario, SCENARIO_LABELS[scenario], COLORS[index], "-")
        for index, scenario in enumerate(["pair_signals", "full_signals"])
    ]
    plot_seed_means(
        axes[1, 1], signal_pairs, "truth_retention_rate", retention_specs, bounded=True
    )
    axes[1, 1].set_title(
        "Selection level: Pairs\nMetric: planted-pair prevalence retention"
    )
    axes[1, 1].set_ylabel("Retention rate")
    axes[1, 1].legend(title="Data scenario", fontsize=8)

    figure.suptitle("Analysis: pair feature selection")
    figure.tight_layout()
    save_figure(figure, "03_pair_selection")


def prediction_figure(prediction):
    target_scenarios = [
        "null_target", "marginal_signals", "pair_signals", "full_signals"
    ]
    data = prediction[
        prediction["scenario"].isin(target_scenarios)
        & prediction["mode"].isin(MODEL_ORDER)
    ].copy()
    present_modes = [mode for mode in MODEL_ORDER if mode in set(data["mode"])]
    specifications = [
        (mode, MODEL_LABELS[mode], COLORS[index], "-")
        for index, mode in enumerate(present_modes)
    ]
    auc = data["roc_auc"].dropna()
    if auc.empty:
        raise ValueError("No AUROC results are available to plot")
    seed_summary = data.groupby(["scenario", "mode", "n"])["roc_auc"].agg(
        ["mean", "std"]
    )
    deviation = seed_summary["std"].fillna(0)
    lower = min(0.5, auc.min(), (seed_summary["mean"] - deviation).min())
    upper = max(0.5, auc.max(), (seed_summary["mean"] + deviation).max())
    padding = max(0.02, 0.08 * (upper - lower))
    limits = max(0, lower - padding), min(1, upper + padding)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, sharey=True)
    for axis, scenario in zip(axes.flat, target_scenarios):
        panel = data[data["scenario"].eq(scenario)].copy()
        panel["series"] = panel["mode"]
        plot_seed_means(axis, panel, "roc_auc", specifications, bounded=True)
        axis.set_ylim(limits)
        axis.set_title(
            f"Data scenario: {SCENARIO_LABELS[scenario]}\nMetric: AUROC"
        )
        axis.set_ylabel("AUROC")
        axis.axhline(0.5, color="black", linestyle=":", linewidth=1)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(
        handles, labels, title="Prediction model", loc="lower center", ncol=3,
        fontsize=9,
    )
    figure.suptitle("Analysis: held-out prediction-model evaluation")
    figure.tight_layout(rect=(0, 0.07, 1, 1))
    save_figure(figure, "04_prediction_quality")
    return present_modes


def main():
    if not RESULTS.exists():
        raise FileNotFoundError(f"Synthetic evaluation results not found: {RESULTS}")
    FIGURES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "axes.grid": True,
        "grid.alpha": 0.25,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    })

    with (PROJECT / "synthetic_config.json").open(encoding="utf-8") as handle:
        preset = json.load(handle)["presets"][PRESET]
    truth_counts = {
        "bacterium": int(preset["bacteria_clusters"]),
        "gene": int(preset["gene_clusters"]),
    }

    clustering = load_runs("clustering_metrics.csv")
    selection = load_runs("selection_metrics.csv")
    prediction = load_runs("prediction_metrics.csv")
    run_count = clustering[["n", "seed", "scenario"]].drop_duplicates().shape[0]

    clustering_figure(clustering, truth_counts)
    marginal_selection_figure(selection)
    pair_selection_figure(selection)
    present_modes = prediction_figure(prediction)

    save_average(
        clustering,
        ["n", "scenario", "scope", "level"],
        ["ari", "nmi", "active_coverage", "recovered_clusters", "singleton_fraction"],
        "clustering_scores.csv",
    )
    save_average(
        selection,
        ["n", "scenario", "level"],
        [
            "precision", "recall", "f1", "false_discovery_proportion",
            "true_positive", "false_positive", "selected",
        ],
        "selection_scores.csv",
    )
    save_average(
        prediction[prediction["mode"].isin(MODEL_ORDER)],
        ["n", "scenario", "mode"],
        ["roc_auc", "average_precision", "total_features"],
        "prediction_scores.csv",
    )

    print(f"Loaded {run_count} runs from {RESULTS}")
    print(f"Prediction modes plotted: {', '.join(present_modes)}")
    if "unclustered_marginals" in set(prediction["mode"]):
        print("Ignored legacy unclustered_marginals results")
    if "pairs_only" not in present_modes:
        print("pairs_only is absent from these saved runs; rerun evaluation after the model change")
    print(f"Saved four PNG figures and averaged score tables to {FIGURES}")


if __name__ == "__main__":
    main()
