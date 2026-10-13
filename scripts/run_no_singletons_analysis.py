#!/usr/bin/env python3
"""Drop baseline singleton clusters BEFORE MW/BH; never reload or rebuild graphs."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
import time
import traceback

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_qinj_redundancy_analysis import evaluate, screen, validate_outer_split, K_VALUES, SEED
from scripts.run_test_free_feature_selection import verify_split, verify_predictors
from scripts.run_posthoc_penalty_search import cached_root
from scripts.plot_aggregate_coverage_comparison import aggregate
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = PROJECT/'outputs/no_singletons_analysis'
REPRESENTATIONS = PROJECT/'outputs/test_free_feature_selection'
DATASETS = ['HMP_2019_ibdmdb_unique', 'QinJ_2012', 'JieZ_2017', 'LiJ_2017',
            'NielsenHB_2014_unique', 'RubelMA_2020', 'WirbelJ_2018', 'ZhuF_2020']
GRIDS = {'fine': list(range(1, 11)), 'coarse': [5, 10, 20, 35, 50, 60, 75, 85, 100]}
METRICS = ['auroc', 'precision', 'recall', 'f1']
METHODS = {'bacteria': 'Bacteria-collapsed | MW/BH',
           'baseline': 'Bernoulli-join mean | MW/BH, all clusters',
           'no_singletons': 'Bernoulli-join mean | remove singletons → MW/BH'}
COLORS = ['#F58518', '#4C78A8', '#54A24B']


def screen_non_singletons(train, test, target, sizes):
    sizes = np.asarray(sizes)
    if (train.shape[1] != len(sizes) or test.shape[1] != len(sizes)
            or np.any(sizes < 1) or not np.issubdtype(sizes.dtype, np.integer)):
        raise ValueError('Invalid original cluster sizes or matrix alignment')
    ids = np.flatnonzero(sizes > 1)
    # Redefine the tested family before BH; do not filter the old significant list.
    t, v, table = screen(train[:, ids], test[:, ids], target, ids, 'pair_cluster', sizes[ids])
    if not table.pair_count.gt(1).all():
        raise ValueError('Singleton survived selection')
    return t, v, table


def choose_k(cv):
    keys = ['method', 'requested_k']
    expected = {(m, k) for m in METHODS for k in K_VALUES}
    grouped = cv.groupby(keys)
    if (cv.duplicated(['fold', *keys]).any() or set(cv.fold) != {0, 1, 2}
            or set(grouped.groups) != expected or not grouped.size().eq(3).all()
            or not np.isfinite(cv.auroc).all()):
        raise ValueError('Incomplete/nonfinite three-fold method/K grid')
    mean = grouped.agg(mean_cv_auc=('auroc', 'mean')).reset_index()
    locked = mean.sort_values(['method', 'mean_cv_auc', 'requested_k'],
        ascending=[True, False, True]).groupby('method').head(1).copy()
    return mean, locked


def prepare_space(dataset, name, target, outer_train, outer_test, expected_a, expected_b):
    source = cached_root(dataset)/name
    a, b, ids = verify_split(source, outer_train, outer_test, len(target))
    if not np.array_equal(a, expected_a) or not np.array_equal(b, expected_b):
        raise ValueError('Cached fold differs from seed-42 stratified split')
    graph = PROJECT/'outputs/method_comparison/real'/dataset/'bernoulli_join_graph' if name == 'outer' else source/'join'
    if name == 'outer':
        validate_outer_split(graph.parent, a, b, ids)
    with np.load(graph/'assignments.npz') as saved:
        labels = saved['clusters']
        if not np.array_equal(saved['feature_ids'], ids):
            raise ValueError('Assignment/prevalence feature IDs differ')
    if (len(labels) != len(ids) or not len(labels) or np.any(labels < 0)
            or not np.issubdtype(labels.dtype, np.integer)):
        raise ValueError('Invalid baseline assignments')
    sizes = np.bincount(labels)
    counts = json.loads((REPRESENTATIONS/dataset/name/'representations_complete.json').read_text())
    integration = json.loads((graph/'integration.json').read_text())
    if (integration['status'] != 'complete' or not counts['bh_predictors_verified']
            or len(sizes) != integration['clusters'] or len(sizes) != counts['clusters']
            or np.any(sizes == 0) or counts['training_samples'] != len(a)
            or counts['validation_samples'] != len(b) or counts['retained_pairs'] != len(ids)
            or counts['training_classes'] != np.bincount(target[a]).tolist()):
        raise ValueError('Invalid full-cluster representation provenance')
    directory = OUTPUT/dataset/name
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'complete.json'
    if marker.exists():
        return (sparse.load_npz(directory/'ranked_train.npz'), sparse.load_npz(directory/'ranked_test.npz'),
                pd.read_csv(directory/'selected_features.csv.gz', float_precision='round_trip').iloc[:100],
                json.loads(marker.read_text()))
    train, test = [sparse.load_npz(REPRESENTATIONS/dataset/name/f'all_join_{suffix}.npz') for suffix in ('train', 'test')]
    if train.shape != (len(a), len(sizes)) or test.shape != (len(b), len(sizes)):
        raise ValueError('Unaligned all-cluster predictors')
    for x in (train, test):
        if not np.isfinite(x.data).all() or np.any(x.data < 0):
            raise ValueError('Invalid abundances')
    verify_predictors(train, test, source)
    t, v, table = screen_non_singletons(train, test, target[a], sizes)
    count = dict(space=name, **counts, removed_singletons=int(np.sum(sizes == 1)),
        tested_non_singleton_clusters=int(np.sum(sizes > 1)), bh_significant_non_singletons=len(table))
    table.to_csv(directory/'selected_features.csv.gz', index=False)
    t, v = t[:, :100], v[:, :100]
    sparse.save_npz(directory/'ranked_train.npz', t)
    sparse.save_npz(directory/'ranked_test.npz', v)
    marker.write_text(json.dumps(count, indent=2))
    print(f'{dataset} {name}: samples={len(a)}/{len(b)}, classes={counts["training_classes"]}, '
          f'pairs={len(ids):,}, bacteria={counts["retained_bacteria"]:,}, genes={counts["retained_genes"]:,}, '
          f'clusters={len(sizes):,}, removed={count["removed_singletons"]:,}, '
          f'tested={count["tested_non_singleton_clusters"]:,}, BH significant={len(table):,}', flush=True)
    return t, v, table.iloc[:100], count


def dataset_run(dataset):
    started = time.monotonic()
    root = cached_root(dataset)
    if (json.loads((root/'complete.json').read_text()).get('status') != 'complete'
            or json.loads((root/'complete.json').read_text())['signature'] != json.loads((root/'protocol.json').read_text())['signature']):
        raise ValueError('Incomplete source experiment')
    samples = pd.read_csv(PROJECT/'outputs/preprocessing'/dataset/'samples.csv')
    y = samples.y.to_numpy(int)
    if (set(np.unique(y)) != {0, 1} or samples.sample_id.duplicated().any()
            or ('subject_id' in samples and samples.subject_id.duplicated().any())):
        raise ValueError('Expected independent samples with binary labels')
    a, b = next(StratifiedShuffleSplit(1, test_size=.2, random_state=SEED).split(np.zeros(len(y)), y))
    directory = OUTPUT/dataset
    directory.mkdir(parents=True, exist_ok=True)
    references = pd.read_csv(REPRESENTATIONS/dataset/'inner_metrics.csv', float_precision='round_trip')
    frames = [references[references.method.isin(('baseline', 'bacteria'))].drop(columns='parameter')]
    counts = []
    for fold, (aa, bb) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(a, y[a])):
        t, v, _, counter = prepare_space(dataset, f'fold_{fold}', y, a, b, a[aa], a[bb])
        counts.append(counter)
        path = directory/f'fold_{fold}'/'metrics.csv'
        if path.exists(): metric = pd.read_csv(path, float_precision='round_trip')
        else:
            metric, _ = evaluate(t, v, y[a[aa]], y[a[bb]], K_VALUES)
            metric.to_csv(path, index=False)
        frames.append(metric.assign(fold=fold, method='no_singletons'))
    cv = pd.concat(frames, ignore_index=True)
    mean, locked = choose_k(cv)
    cv.to_csv(directory/'inner_metrics.csv', index=False)
    mean.to_csv(directory/'mean_cv_by_k.csv', index=False)
    locked.to_csv(directory/'locked_parameters.csv', index=False)
    # All three operating K choices are frozen before accessing the holdout fits.
    train, test, selected, counter = prepare_space(dataset, 'outer', y, a, b, a, b)
    counts.append(counter)
    frames, predictions, tables = [], [], []
    old = pd.read_csv(root/'metrics.csv', float_precision='round_trip')
    for method in METHODS:
        if method == 'no_singletons':
            t, v, table = train, test, selected
        else:
            family = 'join' if method == 'baseline' else 'bacteria'
            t, v = [sparse.load_npz(root/'outer'/f'{family}_{suffix}.npz')[:, :100] for suffix in ('train', 'test')]
            table = pd.read_csv(root/'outer'/f'{family}_features.csv.gz', float_precision='round_trip').iloc[:100]
        metric, prediction = evaluate(t, v, y[a], y[b], K_VALUES)
        if method != 'no_singletons':
            ref = old[old.method.eq(method)].sort_values('requested_k')
            np.testing.assert_allclose(metric[['effective_k', *METRICS]], ref[['effective_k', *METRICS]], rtol=0, atol=1e-12)
        k = int(locked.set_index('method').loc[method, 'requested_k'])
        frames.append(metric.assign(dataset=dataset, method=method, cv_selected_k=k))
        predictions.append(prediction.assign(method=method, sample_index=b[prediction.test_position]))
        tables.append(table.assign(method=method, selected_at_cv_k=table['rank'] <= k))
    metrics = pd.concat(frames, ignore_index=True)
    metrics.to_csv(directory/'holdout_metrics.csv', index=False)
    pd.concat(predictions).to_csv(directory/'holdout_predictions.csv.gz', index=False)
    pd.concat(tables).to_csv(directory/'selected_features.csv.gz', index=False)
    pd.DataFrame(counts).to_csv(directory/'candidate_counts.csv', index=False)
    chosen = metrics[metrics.requested_k.eq(metrics.cv_selected_k)].merge(locked[['method', 'mean_cv_auc']], on='method')
    chosen.to_csv(directory/'cv_selected_models.csv', index=False)
    for grid, ks in GRIDS.items():
        plot_curves(metrics[metrics.requested_k.isin(ks)], dataset, directory/f'{grid}_k_metrics.png', ks)
    (directory/'result.json').write_text(json.dumps(dict(status='complete', references_reproduced=True,
        wall_seconds=time.monotonic()-started), indent=2))
    print(f'COMPLETE {dataset} in {time.monotonic()-started:.1f}s', flush=True)


def plot_curves(data, title, path, ks):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for index, ((method, label), color) in enumerate(zip(METHODS.items(), COLORS)):
            curve = data[data.method.eq(method)].sort_values('requested_k')
            ax.plot(curve.requested_k, curve[metric], color=color, label=label,
                linestyle=['-', '--', ':'][index], marker=['o', 's', '^'][index], linewidth=2, markersize=4)
            if 'cv_selected_k' in curve and metric == 'auroc':
                point = curve[curve.requested_k.eq(curve.cv_selected_k)]
                ax.scatter(point.requested_k, point[metric], color=color, marker='*', s=140, edgecolor='white')
        ax.set(title=metric.upper(), xlabel='Requested K', ylabel=metric.upper(), xticks=ks)
        ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', fontsize=9)
    fig.suptitle(title+'\nSingleton removal before MW/BH; unchanged L2 C=1; exploratory holdout')
    fig.tight_layout(rect=(0, .12, 1, .93))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def summarize():
    complete = [d for d in DATASETS if (OUTPUT/d/'result.json').exists()
                and json.loads((OUTPUT/d/'result.json').read_text())['status'] == 'complete']
    if not complete: return
    data = pd.concat([pd.read_csv(OUTPUT/d/'holdout_metrics.csv') for d in complete], ignore_index=True)
    data.to_csv(OUTPUT/'metrics.csv', index=False)
    counts = pd.concat([pd.read_csv(OUTPUT/d/'candidate_counts.csv').assign(dataset=d) for d in complete], ignore_index=True)
    counts.to_csv(OUTPUT/'candidate_counts.csv', index=False)
    pd.concat([pd.read_csv(OUTPUT/d/'cv_selected_models.csv') for d in complete]).to_csv(OUTPUT/'cv_selected_models.csv', index=False)
    peaks = []
    for grid, ks in {**GRIDS, 'union': K_VALUES}.items():
        subset = data[data.requested_k.isin(ks)].sort_values('requested_k')
        best = subset.loc[subset.groupby(['dataset', 'method']).auroc.idxmax()].copy()
        refs = best.pivot(index='dataset', columns='method', values='auroc')
        best['peak_delta_vs_baseline'] = best.auroc-best.dataset.map(refs.baseline)
        best['peak_gap_to_bacteria'] = best.dataset.map(refs.bacteria)-best.auroc
        peaks.append(best.assign(grid=grid))
        if grid == 'union': continue
        mean, sd = aggregate(subset, complete, ks, methods=METHODS)
        mean.to_csv(OUTPUT/f'{grid}_k_aggregate_mean.csv', index=False)
        sd.to_csv(OUTPUT/f'{grid}_k_aggregate_sd.csv', index=False)
        plot_curves(mean, f'{len(complete)}/8 real datasets: equal-weight mean ({grid} K)', OUTPUT/f'{grid}_k_aggregate_real.png', ks)
        fig, axes = plt.subplots(2, 4, figsize=(18, 9), sharex=True)
        for ax, dataset in zip(axes.flat, DATASETS):
            for index, ((method, label), color) in enumerate(zip(METHODS.items(), COLORS)):
                curve = subset[subset.dataset.eq(dataset) & subset.method.eq(method)]
                ax.plot(curve.requested_k, curve.auroc, label=label, color=color,
                    linestyle=['-', '--', ':'][index], marker=['o', 's', '^'][index], markersize=3)
            ax.set(title=dataset if dataset in complete else dataset+' (pending)', xlabel='Requested K', ylabel='AUROC', xticks=ks)
            ax.tick_params(axis='x', labelsize=7)
            ax.grid(alpha=.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', fontsize=9)
        fig.suptitle(f'Singleton removal before MW/BH: {len(complete)}/8 datasets ({grid} K)')
        fig.tight_layout(rect=(0, .10, 1, .94))
        fig.savefig(OUTPUT/f'{grid}_k_all_real_auc.png', dpi=160)
        plt.close(fig)
    peaks = pd.concat(peaks, ignore_index=True)
    peaks.to_csv(OUTPUT/'peak_summary.csv', index=False)
    table = peaks[peaks.grid.eq('union')].pivot(index='dataset', columns='method', values='auroc')[list(METHODS)]
    lines = ['# Removing singleton Bernoulli-join clusters', '', f'Completed {len(complete)}/8 datasets.', '',
        'Singleton means exactly one original retained bacteria–gene pair. Remove these clusters before running MW and BH over ALL remaining clusters (q≤.10), including clusters not significant in the old family. Constants retain the baseline test behavior. No reclustering, renormalization, bacterial fallback or changed classifier.', '',
        'Reused full, pre-BH continuous sqrt(mean abundance) matrices reconstructed from cached assignments. Each inner fold uses its own training-only preprocessing and cluster sizes. Both outer reference curves were reproduced exactly. Three-fold training CV selects operating K independently for each method; exact ties favor smaller K.', '',
        '## Descriptive peak holdout AUROC over fine + coarse K', '',
        '| Dataset | Bacteria + BH | Join + BH | No singletons + BH |', '|---|---:|---:|---:|']
    lines.extend('| '+d+' | '+' | '.join(f'{v:.3f}' for v in row)+' |' for d, row in table.iterrows())
    lines.extend(['', '`cv_selected_models.csv` reports prediction at training-selected K; `peak_summary.csv` reports descriptive fine/coarse/union maxima. `candidate_counts.csv` includes removed/tested/significant counts for each fold and outer training. Selected-feature tables retain original cluster IDs, sizes, p/q/effect and ranks. Exhausted BH pools report effective K; no discoveries use the unchanged intercept-only fallback.', '',
        'Aggregate figures weight completed datasets equally and explicitly show coverage; partial aggregates are not all-eight results. Previously investigated holdouts remain exploratory.'])
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    (PROJECT/'logs').mkdir(exist_ok=True)
    (PROJECT/'logs/no_singletons_analysis.pid').write_text(str(os.getpid())+'\n')
    print(f'PID {os.getpid()}; sequential datasets, native threads=1', flush=True)
    result = json.loads((REPRESENTATIONS/'result.json').read_text())
    if result['status'] != 'complete' or not result['source_caches_unchanged']:
        raise ValueError('Full pre-BH representation experiment must be complete')
    paths = [Path(__file__), PROJECT/'scripts/run_qinj_redundancy_analysis.py',
             PROJECT/'scripts/run_method_comparison.py', PROJECT/'feature_selection/run_feature_selection.py',
             REPRESENTATIONS/'protocol.json', REPRESENTATIONS/'result.json']
    for dataset in DATASETS:
        root = cached_root(dataset)
        paths.extend([PROJECT/'outputs/preprocessing'/dataset/'samples.csv', root/'complete.json', root/'protocol.json',
                      root/'metrics.csv', REPRESENTATIONS/dataset/'inner_metrics.csv'])
        for name in ('fold_0', 'fold_1', 'fold_2', 'outer'):
            graph = PROJECT/'outputs/method_comparison/real'/dataset/'bernoulli_join_graph' if name == 'outer' else root/name/'join'
            paths.extend([root/name/'split.npz', graph/'assignments.npz', graph/'integration.json',
                          REPRESENTATIONS/dataset/name/'representations_complete.json'])
            paths.extend(REPRESENTATIONS/dataset/name/f'all_join_{s}.npz' for s in ('train', 'test'))
            paths.extend(root/name/f'join_{s}' for s in ('train.npz', 'test.npz', 'features.csv.gz'))
        paths.extend(root/'outer'/f'bacteria_{s}' for s in ('train.npz', 'test.npz', 'features.csv.gz'))
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    protocol = dict(datasets=DATASETS, methods=METHODS, k_values=K_VALUES, seed=SEED,
        selection='Original pair count >1, THEN MW/BH q<=.10 over the remaining full family; no old BH restriction',
        cv='Same seed-42 outer 80/20 and three cached training-only inner folds; choose mean validation AUROC, tie smaller K',
        aggregation='Unchanged sqrt(mean original normalized pair abundances); do not renormalize after removal',
        classifier='Unchanged training-scaled balanced L2 logistic C=1', graph_recomputation=False,
        native_threads=1, holdout='Previously inspected: exploratory')
    signature = hashlib.sha256(repr((states, protocol)).encode()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Changed source/protocol; preserve old outputs and use a new output directory')
    manifest.write_text(json.dumps(dict(signature=signature, source_states=states, **protocol), indent=2))
    status = [dict(dataset=d, status='pending') for d in DATASETS]
    for row in status:
        row['status'] = 'running'
        pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
        marker = OUTPUT/row['dataset']/'result.json'
        try:
            if not marker.exists() or json.loads(marker.read_text()).get('status') != 'complete':
                dataset_run(row['dataset'])
            else: print('Reusing completed '+row['dataset'], flush=True)
            row['status'] = 'complete'
            summarize()
        except Exception as error:
            traceback.print_exc()
            row.update(status='failed', error=str(error))
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(row, indent=2))
        pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
    unchanged = states == [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    complete = all(row['status'] == 'complete' for row in status) and unchanged
    (OUTPUT/'result.json').write_text(json.dumps(dict(status='complete' if complete else 'failed',
        datasets=status, source_caches_unchanged=unchanged, graph_recomputation=False), indent=2))
    if not complete: raise RuntimeError('Incomplete/changed-source experiment; inspect status.csv')
    print('COMPLETE: all eight datasets, three methods, fine/coarse figures', flush=True)


if __name__ == '__main__':
    started = time.monotonic()
    try:
        main()
    finally:
        print(f'Runtime={time.monotonic()-started:.1f}s; peak process RSS='
              f'{resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2:.3f} GiB', flush=True)
