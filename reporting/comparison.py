"""Render only requested outputs from saved experiment tables; never fit methods."""
from dataclasses import asdict
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from execution import cache
from execution.spec import Experiment
from matplotlib.figure import Figure
from matplotlib.axes import Axes

PROJECT = Path(__file__).resolve().parents[1]


def validate(experiment: Experiment) -> None:
    """Require a recognized filename and the metrics needed to produce it."""
    grids = (experiment.evaluation.fine_k, experiment.evaluation.coarse_k)
    required = {f"AUC_{kind}_k_{experiment.evaluation.tag(ks)}.png": {"prediction"}
                for kind in ("average", "overview") for ks in grids}
    required.update({"join_correlation_filter_pilot.png": {"prediction", "correlation"},
        "average_pairwise_absolute_correlation_overview.png": {"correlation"},
        "average_pairwise_absolute_correlation_k_" + experiment.evaluation.tag(grids[1]) + ".png": {"correlation"},
        "negative_log10_p_overview.png": {"shuffled_fdp"}, "real_discoveries_overview.png": {"discoveries"}})
    for name in experiment.outputs:
        if name not in required or not required[name] <= set(experiment.metrics):
            raise ValueError(f"Unknown output or missing required metrics: {name}")


def read_table(directory: Path, name: str) -> pd.DataFrame:
    path = directory / name
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def panels(experiment: Experiment) -> tuple[Figure, np.ndarray]:
    """Return shared-scale dataset panels, hiding unused axes."""
    columns = min(4, len(experiment.datasets))
    fig, axes = plt.subplots((len(experiment.datasets) + columns - 1) // columns, columns,
                             figsize=(6 * columns, 4.5 * ((len(experiment.datasets) + columns - 1) // columns)),
                             squeeze=False, sharey=True)
    for ax in list(axes.flat)[len(experiment.datasets):]:
        ax.set_visible(False)
    return fig, axes


def curves(ax: Axes, table: pd.DataFrame, dataset: str, names: list[str], colors: dict,
           ks: tuple[int, ...], field: str, sd: str | None = None) -> None:
    """Draw one dataset's score-ranked curves and optional descriptive fold SD."""
    part = table[table.dataset.eq(dataset)] if not table.empty else table
    for name in names:
        curve = part[part.method.eq(name) & part.requested_k.isin(ks)].sort_values("requested_k") if not part.empty else part
        if curve.empty:
            continue
        ax.plot(curve.requested_k, curve[field], color=colors[name], marker="o", label=name)
        if sd:
            ax.fill_between(curve.requested_k, curve[field] - curve[sd], curve[field] + curve[sd], color=colors[name], alpha=.08)
    if part.empty or set(part.method) != set(names):
        ax.text(.5, .5, "Incomplete", transform=ax.transAxes, ha="center")
    ax.set(title=dataset.replace("_unique", ""), xlabel="K", xticks=list(ks))
    ax.grid(alpha=.2)


def prepare_reporting_tables(experiment: Experiment, output: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate saved protocols and return prediction, full-data and Top-K tables.

    Writes private summary.csv and p_value_comparison.csv exports; never fits a method.
    """
    tables = output / ".cache"
    definition = tables / "experiment.json"
    if not definition.exists() or json.loads(definition.read_text()) != json.loads(json.dumps(asdict(experiment))):
        raise ValueError("No compatible saved tables; plotting never runs scientific work")
    protocol = tables / "metric_protocol.json"
    if not protocol.exists() or json.loads(protocol.read_text()) != cache.metric_protocol(experiment)[1]:
        raise ValueError("Saved metric definitions changed; regenerate tables from scientific caches before plotting")
    prediction = read_table(tables, "prediction_summary.csv")
    full = read_table(tables, "full_metrics.csv")
    correlation = read_table(tables, "topk_correlations.csv")
    if prediction.empty and full.empty:
        raise ValueError("No completed results available to plot")
    if not full.empty and "log_p_value" in full:
        full.to_csv(tables / "p_value_comparison.csv", index=False)
    if not prediction.empty:
        summary = prediction.merge(full.drop(columns=["effective_k", "pair_count"], errors="ignore"), on=["dataset", "method"], how="left") if not full.empty else prediction
        if not correlation.empty:
            summary = summary.merge(correlation.rename(columns={"average_pairwise_absolute_correlation": "topk_pairwise_correlation",
                "effective_k": "topk_effective_k", "status": "topk_correlation_status"}), on=["dataset", "method", "requested_k"], how="left")
        summary.to_csv(tables / "summary.csv", index=False)
    return prediction, full, correlation


def auc_limits(prediction: pd.DataFrame) -> tuple[float, float]:
    """Shared vertical bounds for dataset panels and K grids."""
    if prediction.empty:
        return 0, 1
    return max(0, prediction.auroc_cv_mean.min() - .04), min(1, prediction.auroc_cv_mean.max() + .04)


def plot_join_filter(experiment: Experiment, prediction: pd.DataFrame, correlation: pd.DataFrame, colors: dict) -> tuple[Figure, str]:
    """Compare join selectors: fine/coarse mean-fold AUROC and score-ranked correlation."""
    names = list(colors)
    fine, coarse = experiment.evaluation.fine_k, experiment.evaluation.coarse_k
    limits = auc_limits(prediction)
    fig, axes = plt.subplots(len(experiment.datasets), 3, figsize=(17, 4.5 * len(experiment.datasets)), squeeze=False)
    for row, dataset in enumerate(experiment.datasets):
        for column, ks in enumerate((fine, coarse)):
            curves(axes[row, column], prediction, dataset, names, colors, ks, "auroc_cv_mean", "auroc_fold_sd")
            axes[row, column].set(ylabel="Mean fold AUROC", ylim=limits)
        curves(axes[row, 2], correlation, dataset, names, colors, coarse, "average_pairwise_absolute_correlation")
        axes[row, 2].set(ylabel="Average pairwise |correlation|", ylim=(0, 1.01))
    title = f"Join correlation filtering\nUnscreened Top-K; {experiment.evaluation.folds}-fold mean ± SD; full-data correlations"
    return fig, title


def plot_auc(experiment: Experiment, prediction: pd.DataFrame, colors: dict, filename: str) -> tuple[Figure, str]:
    """Plot dataset CV curves or their equal-dataset-weight average."""
    names = list(colors)
    fine, coarse = experiment.evaluation.fine_k, experiment.evaluation.coarse_k
    limits = auc_limits(prediction)
    ks = fine if filename.endswith("_" + experiment.evaluation.tag(fine) + ".png") else coarse
    if "average" in filename:
        fig, ax = plt.subplots(figsize=(10, 6))
        complete = [d for d in experiment.datasets if not prediction.empty and set(prediction[prediction.dataset.eq(d)].method) == set(names)]
        data = prediction[prediction.dataset.isin(complete)] if not prediction.empty else prediction
        data = data.groupby(["method", "requested_k"]).auroc_cv_mean.mean().reset_index() if not data.empty else data
        for name in names:
            part = data[data.method.eq(name) & data.requested_k.isin(ks)].sort_values("requested_k") if not data.empty else data
            ax.plot(part.requested_k if not part.empty else [], part.auroc_cv_mean if not part.empty else [], color=colors[name], marker="o", label=name)
        ax.set(xlabel="K", ylabel="AUROC", xticks=list(ks), ylim=limits)
        title = f"Average AUROC; {len(complete)} complete datasets\nEqual dataset weights; unscreened Top-K"
    else:
        fig, axes = panels(experiment)
        for ax, dataset in zip(axes.flat, experiment.datasets):
            curves(ax, prediction, dataset, names, colors, ks, "auroc_cv_mean", "auroc_fold_sd")
            ax.set(ylabel="AUROC", ylim=limits)
        title = f"AUROC; {experiment.evaluation.folds}-fold CV\nMean ± fold SD; unscreened Top-K"
    return fig, title


def plot_statistics(experiment: Experiment, full: pd.DataFrame, correlation: pd.DataFrame, colors: dict, filename: str) -> tuple[Figure, str]:
    """Plot full-data discoveries, shuffled-test p-values, or feature correlation.

    Full significant sets and score-ranked Top-K subsets remain distinct.
    """
    names = list(colors)
    coarse = experiment.evaluation.coarse_k
    fig, axes = panels(experiment)
    topk = filename.startswith("average_pairwise_absolute_correlation_k_")
    p_value = filename == "negative_log10_p_overview.png"
    discoveries = filename == "real_discoveries_overview.png"
    discovery_floor = 1e-6
    if discoveries and not full.empty:
        fractions = full.real_discoveries / full.tested_features.replace(0, np.nan)
        discovery_floor = fractions[fractions > 0].min() / 10 if (fractions > 0).any() else 1e-6
    if not topk and not p_value and not discoveries and filename != "average_pairwise_absolute_correlation_overview.png":
        raise ValueError(f"Unknown requested output: {filename}")
    for ax, dataset in zip(axes.flat, experiment.datasets):
        if topk:
            curves(ax, correlation, dataset, names, colors, coarse, "average_pairwise_absolute_correlation")
            ax.set(ylabel="Average pairwise |correlation|", ylim=(0, 1.01))
            continue
        part = full[full.dataset.eq(dataset)] if not full.empty else full
        for i, name in enumerate(names):
            rows = part[part.method.eq(name)] if not part.empty else part
            if rows.empty:
                ax.text(i, .6, "missing", rotation=90, ha="center", transform=ax.get_xaxis_transform())
                continue
            row = rows.iloc[0]
            if p_value:
                ax.bar(i, -row.normal_log_p_value / np.log(10), color=colors[name], alpha=.25, width=.8)
                ax.bar(i, -row.log_p_value / np.log(10), color=colors[name], width=.45)
            elif discoveries:
                fraction = row.real_discoveries / row.tested_features if row.tested_features else np.nan
                ax.bar(i, fraction, color=colors[name])
                # Zero counts are labeled at the axis floor, without a positive bar.
                ax.annotate(f"{int(row.real_discoveries):,} / {int(row.tested_features):,}",
                            (i, fraction if fraction > 0 else discovery_floor),
                            xytext=(0, 4), textcoords="offset points", ha="center", fontsize=7)
            elif np.isfinite(row.average_pairwise_absolute_correlation):
                ax.bar(i, row.average_pairwise_absolute_correlation, color=colors[name])
            else:
                ax.text(i, .8, row.status, rotation=90, ha="center", transform=ax.get_xaxis_transform(), fontsize=8)
        ax.set(title=dataset.replace("_unique", ""), xticks=range(len(names)), xticklabels=names)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="y", alpha=.2)
        if p_value:
            ax.set_yscale("symlog", linthresh=3)
            ax.axhline(-np.log10(.05), color="black", linestyle=":")
        elif discoveries:
            ax.set_yscale("log")
            ax.set(ylabel="Discoveries / tested features (log scale)", ylim=(discovery_floor, 1))
        else:
            ax.set_ylim(0, 1.01)
    title = (f"−log10(p-value); {experiment.evaluation.shuffles} shuffles\nPermutation vs Binomial/Normal approximation"
             if p_value else "Discovery probability; full datasets" if discoveries else "Average pairwise |correlation|; full datasets")
    return fig, title


def render(experiment: Experiment, *, output: Path | None = None) -> None:
    """Render requested figures from compatible saved tables; never fit methods."""
    validate(experiment)
    output = PROJECT / "outputs" / experiment.name if output is None else Path(output)
    prediction, full, correlation = prepare_reporting_tables(experiment, output)
    palette = {"Bacteria-collapsed": "tab:orange", "Gene-collapsed": "tab:green", "Bernoulli-join": "tab:blue"}
    colors = {method.name: palette.get(method.name, plt.get_cmap("tab10")(i % 10))
              for i, method in enumerate(experiment.methods)}
    names = list(colors)
    for filename in experiment.outputs:
        if filename == "join_correlation_filter_pilot.png":
            fig, title = plot_join_filter(experiment, prediction, correlation, colors)
        elif filename.startswith("AUC_"):
            fig, title = plot_auc(experiment, prediction, colors, filename)
        else:
            fig, title = plot_statistics(experiment, full, correlation, colors, filename)
        fig.suptitle(title)
        if filename == "negative_log10_p_overview.png":
            fig.legend(handles=[Patch(facecolor="gray", label="Permutation (solid foreground)"), Patch(facecolor="gray", alpha=.25, label="Binomial/Normal (transparent)")], loc="lower center", ncol=2)
            fig.text(.5, .015, "Global-null count test, not an FDP/FDR certificate; Normal reference is model-dependent", ha="center", fontsize=9)
        else:
            fig.legend(handles=[Line2D([], [], color=colors[n], label=n) for n in names], loc="lower center", ncol=min(4, len(names)))
        fig.tight_layout(rect=(0, .08, 1, .92))
        fig.savefig(output / filename, dpi=150)
        plt.close(fig)
