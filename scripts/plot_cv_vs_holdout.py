#!/usr/bin/env python3
"""Plot cached mean-fold CV and historical 80/20 AUROC; never fit models."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
INPUT = PROJECT / 'outputs/cv_evaluation/all_real'
OUTPUT = INPUT / 'figures'
DATASETS = ['HMP_2019_ibdmdb_unique', 'QinJ_2012', 'JieZ_2017', 'LiJ_2017',
            'NielsenHB_2014_unique', 'RubelMA_2020', 'WirbelJ_2018', 'ZhuF_2020']
LABELS = {'bacteria': 'Bacteria-collapsed', 'gene': 'Gene-collapsed',
          'join': 'Bernoulli-join mean'}
COLORS = {'bacteria': '#F58518', 'gene': '#B279A2', 'join': '#4C78A8'}
GRIDS = {'fine': list(range(1, 11)), 'coarse': [5, 10, 20, 35, 50, 60, 75, 85, 100]}
STYLES = {'Mean 5-fold CV': '-', 'Old 80/20': ':'}


def comparison_table(cv, holdout, completed):
    expected_k = set(GRIDS['fine']) | set(GRIDS['coarse'])
    if not set(completed) <= set(DATASETS):
        raise ValueError('Unrecognized completed datasets')
    tables = []
    for table, evaluation, datasets in [(cv, 'Mean 5-fold CV', completed),
                                        (holdout, 'Old 80/20', DATASETS)]:
        table = table[['dataset', 'method', 'requested_k', 'auroc']].copy()
        if set(table.dataset) != set(datasets) or set(table.method) != set(LABELS):
            raise ValueError(f'Unexpected dataset/method coverage: {evaluation}')
        if table.duplicated(['dataset', 'method', 'requested_k']).any():
            raise ValueError(f'Duplicate curve points: {evaluation}')
        if not np.isfinite(table.auroc).all() or not table.auroc.between(0, 1).all():
            raise ValueError(f'Invalid AUROC: {evaluation}')
        if evaluation == 'Mean 5-fold CV' and not cv.n_folds.eq(5).all():
            raise ValueError('CV curves require five completed folds')
        for dataset in datasets:
            for method in LABELS:
                curve = table[table.dataset.eq(dataset) & table.method.eq(method)]
                if set(curve.requested_k) != expected_k:
                    raise ValueError(f'Incomplete K grid: {evaluation}/{dataset}/{method}')
        table['evaluation'] = evaluation
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def plot_comparison(table, grid):
    k_values = GRIDS[grid]
    figure, axes = plt.subplots(2, 4, figsize=(19, 10), sharex=True, sharey=True)
    cv_datasets = set(table.loc[table.evaluation.eq('Mean 5-fold CV'), 'dataset'])
    # Use the full fine/coarse union so both figures share one tight scale.
    limits = (max(0, np.floor((table.auroc.min() - .01) * 100) / 100),
              min(1.01, np.ceil((table.auroc.max() + .01) * 100) / 100))
    for axis, dataset in zip(axes.flat, DATASETS):
        selected = table[table.dataset.eq(dataset) & table.requested_k.isin(k_values)]
        for method in LABELS:
            for evaluation, style in STYLES.items():
                curve = selected[selected.method.eq(method) & selected.evaluation.eq(evaluation)]
                if curve.empty:
                    continue
                curve = curve.sort_values('requested_k')
                axis.plot(curve.requested_k, curve.auroc, color=COLORS[method],
                          linestyle=style, linewidth=2.2, marker='o', markersize=3,
                          label=f'{LABELS[method]} — {evaluation}')
        axis.set(title=dataset.replace('_unique', ''), xlabel='Requested K',
                 ylabel='AUROC', xticks=k_values, ylim=limits)
        axis.tick_params(axis='x', labelsize=9, rotation=45 if grid == 'coarse' else 0)
        axis.grid(alpha=.2)
        if dataset not in cv_datasets:
            axis.text(.5, .08, 'CV unavailable\n(5-fold evaluation incomplete)',
                      transform=axis.transAxes, ha='center', va='bottom', fontsize=10,
                      bbox=dict(facecolor='white', edgecolor='none', alpha=.9))
    handles = [Line2D([], [], color=COLORS[method], linestyle=style, linewidth=2.2,
                      label=f'{LABELS[method]} — {evaluation}')
               for evaluation, style in STYLES.items() for method in LABELS]
    figure.legend(handles=handles, loc='lower center', ncol=3, fontsize=11,
                  bbox_to_anchor=(.5, .02), frameon=False)
    figure.suptitle(f'Real datasets: mean 5-fold CV vs old 80/20 — {grid} K', fontsize=18, y=.98)
    figure.text(.5, .935,
                f'Solid: mean fold AUROC  |  Dotted: single holdout AUROC  |  '
                f'CV complete: {len(cv_datasets)}/{len(DATASETS)} datasets',
                ha='center', fontsize=12)
    figure.tight_layout(rect=(0, .12, 1, .91))
    return figure, axes


def main():
    completed = json.loads((INPUT / 'result.json').read_text())['completed_datasets']
    table = comparison_table(pd.read_csv(INPUT / 'mean_fold_metrics.csv'),
                             pd.read_csv(INPUT / 'historical_holdout_metrics.csv'), completed)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUTPUT / 'cv_vs_80_20_plot_metrics.csv', index=False)
    for grid in GRIDS:
        figure, _ = plot_comparison(table, grid)
        path = OUTPUT / f'{grid}_k_cv_vs_80_20.png'
        figure.savefig(path, dpi=180, bbox_inches='tight')
        plt.close(figure)
        print(f'Saved {path}', flush=True)


if __name__ == '__main__':
    main()
