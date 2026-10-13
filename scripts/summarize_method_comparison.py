#!/usr/bin/env python3
"""Three-method real-dataset tables and figures from cached 80/20 results."""
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scripts.run_method_comparison import METHOD_LABELS, REAL_LABELS, K_VALUES

ROOT = PROJECT/'outputs/method_comparison'
FIGURES, TABLES = ROOT/'figures', ROOT/'tables'
METHOD_ORDER = list(METHOD_LABELS)
METRICS = ['auroc', 'precision', 'recall', 'f1']
COLORS = dict(zip(METHOD_ORDER, ['#F58518', '#B279A2', '#4C78A8']))


def data_limits(values):
    values = np.asarray(values, dtype=float)
    return max(0., np.nanmin(values)-.04), min(1.02, np.nanmax(values)+.04)


def plot_curves(curves, title, name, errors=None):
    FIGURES.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for axis, metric in zip(axes.flat, METRICS):
        limits = curves[metric]
        if errors is not None:
            bounds = curves[['method', 'requested_k', metric]].merge(
                errors[['method', 'requested_k', metric]], on=['method', 'requested_k'],
                suffixes=('', '_sd'), validate='one_to_one')
            spread = bounds[metric+'_sd'].fillna(0)
            limits = np.r_[bounds[metric]-spread, bounds[metric]+spread]
        for method in METHOD_ORDER:
            curve = curves[curves.method.eq(method)].sort_values('requested_k')
            if curve.empty: continue
            axis.plot(curve.requested_k, curve[metric], color=COLORS[method],
                      marker='o', markersize=4, linewidth=2, label=METHOD_LABELS[method])
            if errors is not None:
                spread = errors[errors.method.eq(method)].set_index('requested_k').loc[curve.requested_k, metric].fillna(0).to_numpy()
                axis.fill_between(curve.requested_k, np.clip(curve[metric]-spread, 0, 1),
                                  np.clip(curve[metric]+spread, 0, 1), color=COLORS[method], alpha=.1)
        axis.set(title=metric.upper(), xlabel='Requested K', ylabel=metric.upper(),
                 xticks=sorted(curves.requested_k.unique()), ylim=data_limits(limits))
        axis.grid(alpha=.2)
    figure.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', fontsize=10)
    figure.suptitle(title+'\nExploratory single 80/20 holdout; peaks over K are descriptive')
    figure.tight_layout(rect=(0, .09, 1, .92))
    figure.savefig(FIGURES/f'{name}.png', dpi=160)
    plt.close(figure)


def read_cached(filename):
    tables = []
    for dataset in REAL_LABELS:
        for method in METHOD_ORDER:
            path = ROOT/'real'/dataset/method/filename
            if not path.exists(): raise ValueError(f'Missing completed method output: {path}')
            table = pd.read_csv(path, float_precision='round_trip')
            table['source'], table['dataset'], table['method'] = 'real', dataset, method
            table['method_label'] = METHOD_LABELS[method]
            tables.append(table)
    return pd.concat(tables, ignore_index=True)


def summarize():
    TABLES.mkdir(parents=True, exist_ok=True)
    metrics = read_cached('metrics.csv')
    keys = ['dataset', 'method', 'requested_k']
    expected = {(d, m, k) for d in REAL_LABELS for m in METHOD_ORDER for k in K_VALUES}
    if (metrics.duplicated(keys).any() or set(map(tuple, metrics[keys].to_numpy())) != expected
            or not np.isfinite(metrics[METRICS]).all().all()):
        raise ValueError('Incomplete or duplicate three-method comparison')
    metrics.to_csv(ROOT/'metrics_by_k.csv', index=False, float_format='%.8g')
    summary = metrics.groupby(['dataset', 'method'])[METRICS].max().reset_index()
    summary.to_csv(ROOT/'dataset_summary.csv', index=False)
    summary.groupby('method')[METRICS].mean().to_csv(ROOT/'method_summary.csv')
    for dataset, group in metrics.groupby('dataset'):
        group.to_csv(TABLES/f'real_{dataset}.csv', index=False)
        plot_curves(group, f'{dataset}: 80/20 holdout (coarse K)', f'real_{dataset}')
    grouped = metrics.groupby(['method', 'requested_k'])[METRICS]
    mean, spread = grouped.mean().reset_index(), grouped.std().reset_index()
    mean.assign(n_datasets=len(REAL_LABELS)).to_csv(TABLES/'coarse_k_aggregate_real.csv', index=False)
    plot_curves(mean, 'Real datasets: equal-weight mean ± between-dataset SD (coarse K)',
                'aggregate_real', spread)
    print(f'Saved three-method summaries for {len(REAL_LABELS)} datasets', flush=True)


if __name__ == '__main__':
    summarize()
