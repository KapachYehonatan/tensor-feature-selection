#!/usr/bin/env python3
"""Three clustering families on fixed fold-specific Bernoulli graphs, MW/BH unchanged."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.run_qinj_redundancy_analysis import evaluate, screen, validate_outer_split, standardized, K_VALUES, SEED
from scripts.run_bernoulli_correlation_join import aggregate_pair_clusters, JOIN_PYTHON
from scripts.run_method_comparison import collapsed_average
from scripts.bernoulli_clustering_worker import RESOLUTIONS, MIN_EXTRA, MAX_EXTRA, BATCH, PATIENCE, QUALITY_TOL, LEIDEN_SEED, WORKERS
from research_utils import normalize_rows, prevalence_mask
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = PROJECT/'outputs/bernoulli_clustering_analysis'
DATASETS = ['QinJ_2012', 'HMP_2019_ibdmdb_unique', 'JieZ_2017', 'LiJ_2017',
            'NielsenHB_2014_unique', 'RubelMA_2020', 'WirbelJ_2018', 'ZhuF_2020']
PILOTS = DATASETS[:2]
GRIDS = {'fine': list(range(1, 11)), 'coarse': [5, 10, 20, 35, 50, 60, 75, 85, 100]}
METRICS = ['auroc', 'precision', 'recall', 'f1']
METHODS = {'bacteria': 'Bacteria-collapsed | MW/BH',
    'baseline': 'Bernoulli-join mean: cached Leiden | MW/BH',
    'extended': 'Bernoulli-join mean: extended Leiden | MW/BH',
    'lower_resolution': 'Bernoulli-join mean: lower-resolution Leiden | MW/BH',
    'hierarchy': 'Bernoulli-join mean: Leiden → hierarchical merging | MW/BH'}
COLORS = ['#F58518', '#4C78A8', '#54A24B', '#B279A2', '#E45756']
WORKER = PROJECT/'scripts/bernoulli_clustering_worker.py'


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def cached_root(dataset):
    return PROJECT/'outputs/qinj_redundancy_analysis' if dataset == 'QinJ_2012' else PROJECT/'outputs/real_redundancy_analysis'/dataset


def graph_path(root):
    integration = json.loads((root/'integration.json').read_text())
    threshold = float(integration['effective_threshold'])
    original = root/'graph'
    source = original if float(json.loads((original/'run.json').read_text())['threshold']) == threshold else root/f'graph-threshold-{threshold:.2f}'
    manifest = json.loads((source/'run.json').read_text())
    if integration['status'] != 'complete' or manifest['status'] != 'complete' or float(manifest['threshold']) != threshold or not (source/'edges').is_dir():
        raise ValueError(f'Unavailable fixed graph: {root}; graph rebuilding is forbidden')
    return source, integration


def settings():
    rows = [dict(method='bacteria', setting='bacteria', order=0),
            dict(method='baseline', setting='baseline', order=0),
            dict(method='extended', setting='resolution_1', order=0)]
    rows.extend(dict(method='lower_resolution', setting=f'resolution_{r:g}', resolution=r, order=i)
                for i, r in enumerate(RESOLUTIONS[1:]))
    rows.extend(dict(method='hierarchy', setting=f'{link}_cut_{cut}', linkage=link, cut=cut, order=i*3+j)
                for i, link in enumerate(('average', 'complete')) for j, cut in enumerate(('baseline', '0.9', '0.95')))
    return rows


def choose_settings(cv):
    expected = {(s['method'], s['setting'], k) for s in settings() for k in K_VALUES}
    keys = ['method', 'setting', 'requested_k']
    grouped = cv.groupby(keys)
    if cv.duplicated(['fold', *keys]).any() or set(cv.fold) != {0, 1, 2} or set(grouped.groups) != expected or not grouped.size().eq(3).all() or not np.isfinite(cv.auroc).all():
        raise ValueError('Incomplete/nonfinite three-fold clustering/K grid')
    mean = grouped.agg(mean_cv_auc=('auroc', 'mean'), mean_effective_k=('effective_k', 'mean')).reset_index()
    order = {s['setting']: s['order'] for s in settings()}
    mean['parameter_order'] = mean.setting.map(order)
    locked = mean.sort_values(['method', 'mean_cv_auc', 'parameter_order', 'requested_k'],
        ascending=[True, False, True, True]).groupby('method').head(1).copy()
    return mean, locked


def verify_split(source, expected_train, expected_test, outer_test, outer=False):
    with np.load(source/'split.npz') as saved:
        a, b, ids = saved['train_index'], saved['test_index'], saved['feature_ids']
    if not np.array_equal(a, expected_train) or not np.array_equal(b, expected_test):
        raise ValueError(f'Cached split does not match deterministic fold: {source}')
    if np.intersect1d(a, b).size or len(np.unique(a)) != len(a) or len(np.unique(b)) != len(b):
        raise ValueError('Invalid overlapping/duplicate sample split')
    if not outer and np.intersect1d(np.r_[a, b], outer_test).size:
        raise ValueError('Outer holdout entered inner graph/preprocessing')
    return ids


def nearest_diagnostics(x):
    """Nearest neighbor means maximum absolute r, with its corresponding sign."""
    z, valid = standardized(x[:, :10])
    correlation = z.T@z
    correlation[~valid, :] = np.nan
    correlation[:, ~valid] = np.nan
    result = dict(top_n=x[:, :10].shape[1], nearest_abs_median=np.nan, nearest_abs_max=np.nan,
                  nearest_signed_median=np.nan, high_correlation_pairs=0, effective_rank=np.nan)
    if valid.any():
        eig = np.maximum(np.linalg.eigvalsh(correlation[np.ix_(valid, valid)]), 0.)
        prob = eig[eig > 1e-12]/eig.sum()
        result['effective_rank'] = float(np.exp(-np.sum(prob*np.log(prob))))
    neighbors = []
    if valid.sum() >= 2:
        off = correlation.copy()
        np.fill_diagonal(off, np.nan)
        for i in np.flatnonzero(valid):
            j = int(np.nanargmax(np.abs(off[i])))
            neighbors.append(dict(feature_position=i, neighbor_position=j, signed_r=float(off[i, j]), abs_r=float(abs(off[i, j]))))
        values = np.array([r['signed_r'] for r in neighbors])
        result.update(nearest_abs_median=float(np.median(np.abs(values))), nearest_abs_max=float(np.max(np.abs(values))),
            nearest_signed_median=float(np.median(values)), high_correlation_pairs=int(np.sum(np.abs(off[np.triu_indices(len(off), 1)]) >= .8)))
    return result, correlation, neighbors


def partition_summary(labels, baseline):
    sizes = np.bincount(labels)
    if np.any(sizes == 0) or len(labels) != len(baseline):
        raise ValueError('Invalid cluster labels')
    # Count split/merged communities without quadratic contingency matrices.
    edges = np.unique(np.column_stack((baseline, labels)), axis=0)
    return dict(clusters=len(sizes), singleton_clusters=int(np.sum(sizes == 1)),
        singleton_fraction=float(np.mean(sizes == 1)), largest_cluster=int(sizes.max()),
        size_median=float(np.median(sizes)), size_p90=float(np.quantile(sizes, .9)), size_p99=float(np.quantile(sizes, .99)),
        ari_vs_baseline=float(adjusted_rand_score(baseline, labels)),
        baseline_clusters_split=int(np.sum(np.bincount(edges[:, 0]) > 1)),
        resulting_clusters_merging_baseline=int(np.sum(np.bincount(edges[:, 1]) > 1)))


def clustering(dataset, name, graph_root, output):
    graph, integration = graph_path(graph_root)
    hierarchy = [dict(key=s['setting'], linkage=s['linkage'], cutoff=float(integration['effective_threshold']) if s['cut'] == 'baseline' else float(s['cut']))
                 for s in settings() if s['method'] == 'hierarchy']
    # JOIN_PYTHON is <source>/.venv/bin/python: the source is three parents up.
    inputs = [WORKER, JOIN_PYTHON.parents[2]/'community_detection.py', JOIN_PYTHON.parents[2]/'bernoulli_join.py',
              graph_root/'assignments.npz', graph_root/'integration.json', graph/'run.json', graph/'nodes.npy',
              *sorted((graph/'edges').glob('part-*.npy'))]
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]
    signature = hashlib.sha256(repr((states, hierarchy)).encode()).hexdigest()
    request = dict(graph=str(graph), baseline=str(graph_root/'assignments.npz'), output=str(output),
                   hierarchy=hierarchy, signature=signature, dataset=dataset, space=name)
    output.mkdir(parents=True, exist_ok=True)
    marker = output/'complete.json'
    if marker.exists() and json.loads(marker.read_text())['signature'] != signature:
        raise ValueError('Changed fixed graph/worker inputs; preserve caches and use a new directory')
    if not marker.exists():
        write_json(output/'request.json', request)
        print(f'{dataset} {name}: clustering fixed phi={integration["effective_threshold"]:.3f}', flush=True)
        subprocess.run([str(JOIN_PYTHON), '-u', str(WORKER), str(output/'request.json')], check=True)
    info = json.loads(marker.read_text())
    if not all(info['converged'].values()):
        raise RuntimeError(f'Nonconverged Leiden settings: {info["converged"]}; no substitution')
    return integration


def screen_cached(train, validation, target, labels, directory):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'complete.json'
    if marker.exists():
        return (sparse.load_npz(directory/'ranked_train.npz'), sparse.load_npz(directory/'ranked_test.npz'),
                pd.read_csv(directory/'features.csv.gz', float_precision='round_trip'))
    t, v, sizes = aggregate_pair_clusters(train, validation, labels)
    t, v, table = screen(t, v, target, np.arange(len(sizes)), 'pair_cluster', sizes)
    screened_count = len(table)
    # Whole family was tested; cache only up to 100 screened predictors needed here.
    t, v, table = t[:, :100], v[:, :100], table.iloc[:100].copy()
    sparse.save_npz(directory/'ranked_train.npz', t)
    sparse.save_npz(directory/'ranked_test.npz', v)
    table.to_csv(directory/'features.csv.gz', index=False)
    write_json(marker, dict(status='complete', full_family_clusters=len(sizes), bh_significant_clusters=screened_count, screened_cache_count=len(table)))
    return t, v, table


def compare_reference(t, v, table, source, family):
    old_table = pd.read_csv(source/f'{family}_features.csv.gz', float_precision='round_trip').iloc[:100]
    if not np.array_equal(table.feature_id.to_numpy(), old_table.feature_id.to_numpy()):
        raise ValueError(f'Baseline BH order changed: {source} {family}')
    for rebuilt, suffix in ((t, 'train'), (v, 'test')):
        old = sparse.load_npz(source/f'{family}_{suffix}.npz')[:, :100]
        if rebuilt.shape != old.shape:
            raise ValueError(f'Baseline reconstruction shape changed: {source} {family}')
        difference = rebuilt-old
        if difference.nnz and np.abs(difference.data).max() > 1e-12:
            raise ValueError(f'Baseline reconstruction changed: {source} {family}')


def prepare_space(matrix, y, pb, pg, dataset, name, a, b, outer_test, selected=None):
    source = cached_root(dataset)/name
    ids = verify_split(source, a, b, outer_test, outer=name == 'outer')
    mask, _, cutoff = prevalence_mask(matrix[a], .05, 5)
    if not np.array_equal(np.flatnonzero(mask), ids):
        raise ValueError('Training-only prevalence differs from cached graph')
    train, validation = normalize_rows(matrix[a][:, ids]), normalize_rows(matrix[b][:, ids])
    graph_root = PROJECT/'outputs/method_comparison/real'/dataset/'bernoulli_join_graph' if name == 'outer' else source/'join'
    if name == 'outer':
        validate_outer_split(graph_root.parent, a, b, ids)
    with np.load(graph_root/'assignments.npz') as saved:
        if not np.array_equal(saved['feature_ids'], ids):
            raise ValueError('Clustering feature IDs differ from fold prevalence IDs')
        baseline = saved['clusters']
    output = OUTPUT/dataset/name
    integration = clustering(dataset, name, graph_root, output/'clustering')
    print(f'{dataset} {name}: samples={len(a)}/{len(b)}, classes={np.bincount(y[a]).tolist()}, '
          f'pairs={len(ids):,}, bacteria={len(np.unique(pb[ids])):,}, genes={len(np.unique(pg[ids])):,}, baseline_clusters={integration["clusters"]:,}', flush=True)
    candidates = settings() if selected is None else selected
    candidates = [*candidates, dict(method='fresh_control', setting='fresh_control', order=0)]
    frames, summaries = [], []
    for setting in candidates:
        key, method = setting['setting'], setting['method']
        directory = output/key
        if method == 'bacteria':
            t = sparse.load_npz(source/'bacteria_train.npz')[:, :100]
            v = sparse.load_npz(source/'bacteria_test.npz')[:, :100]
            table = pd.read_csv(source/'bacteria_features.csv.gz', float_precision='round_trip').iloc[:100]
            # Independently reconstruct the cached reference, training-only.
            bt, bacteria_ids = collapsed_average(train, pb[ids], int(pb.max())+1)
            bv, _ = collapsed_average(validation, pb[ids], int(pb.max())+1)
            rt, rv, rtable = screen(bt, bv, y[a], bacteria_ids, 'bacterium', np.bincount(pb[ids])[bacteria_ids])
            compare_reference(rt[:, :100], rv[:, :100], rtable.iloc[:100], source, 'bacteria')
            summary = dict(clusters=len(bacteria_ids), bh_significant_clusters=len(rtable),
                           singleton_fraction=float(np.mean(np.bincount(pb[ids])[bacteria_ids] == 1)))
            labels = None
        else:
            if method == 'baseline':
                labels = baseline
            else:
                with np.load(output/'clustering'/f'{key}.npz') as saved:
                    if not np.array_equal(saved['feature_ids'], ids):
                        raise ValueError('New clustering assignment IDs changed')
                    labels = saved['clusters']
            t, v, table = screen_cached(train, validation, y[a], labels, directory)
            summary = partition_summary(labels, baseline)
            summary['bh_significant_clusters'] = json.loads((directory/'complete.json').read_text())['bh_significant_clusters']
            if method == 'baseline':
                compare_reference(t, v, table, source, 'join')
        directory.mkdir(parents=True, exist_ok=True)
        table.to_csv(directory/'features.csv.gz', index=False)
        metric_path = directory/'metrics.csv'
        if metric_path.exists():
            metric = pd.read_csv(metric_path, float_precision='round_trip')
        else:
            metric, predictions = evaluate(t, v, y[a], y[b], K_VALUES)
            metric.to_csv(metric_path, index=False)
            if name == 'outer':
                predictions.assign(sample_index=b[predictions.test_position]).to_csv(directory/'predictions.csv.gz', index=False)
        frames.append(metric.assign(method=method, setting=key, space=name))
        diagnostic, correlation, neighbors = nearest_diagnostics(t)
        pd.DataFrame(neighbors, columns=['feature_position', 'neighbor_position', 'signed_r', 'abs_r']).to_csv(directory/'top10_neighbors.csv', index=False)
        pd.DataFrame(correlation).to_csv(directory/'top10_correlation.csv', index=False)
        summaries.append(dict(method=method, setting=key, space=name, retained_pairs=len(ids),
            prevalence_cutoff=int(cutoff), threshold=integration['effective_threshold'], **summary, **diagnostic))
        if name == 'outer':
            fig, ax = plt.subplots(figsize=(6, 5))
            if len(correlation):
                heatmap = ax.imshow(correlation, vmin=-1, vmax=1, cmap='coolwarm')
                fig.colorbar(heatmap, ax=ax, label='Training Pearson r')
            ax.set(title=f'{dataset}\n{method}: within selected Top 10', xlabel='Selected feature position', ylabel='Selected feature position')
            fig.tight_layout()
            fig.savefig(directory/'top10_correlation.png', dpi=160)
            plt.close(fig)
            feature_rows = []
            for rank, row in enumerate(table.itertuples(), 1):
                members = np.flatnonzero(pb[ids] == int(row.feature_id)) if labels is None else np.flatnonzero(labels == int(row.feature_id))
                x = t[:, rank-1]
                feature_rows.append(dict(rank=rank, cluster_id=int(row.feature_id), p=float(row.p), q=float(row.q),
                    pair_count=len(members), training_prevalence=float(x.getnnz()/len(a)),
                    constituent_bacteria=';'.join(map(str, np.unique(pb[ids[members]]))),
                    constituent_baseline_clusters='' if labels is None else ';'.join(map(str, np.unique(baseline[members])))))
            pd.DataFrame(feature_rows, columns=['rank', 'cluster_id', 'p', 'q', 'pair_count', 'training_prevalence',
                'constituent_bacteria', 'constituent_baseline_clusters']).to_csv(directory/'selected_features.csv.gz', index=False)
        print(f'  {dataset} {name} {key}: clusters={summary["clusters"]:,}, effective Top100={t.shape[1]}', flush=True)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(summaries)


def plot_curves(data, title, path, ks, errors=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for index, ((method, label), color) in enumerate(zip(METHODS.items(), COLORS)):
            curve = data[data.method.eq(method)].sort_values('requested_k')
            ax.plot(curve.requested_k, curve[metric], label=label, color=color,
                linestyle=['-', '--', ':', '-.', '--'][index], marker=['o', 's', '^', 'D', 'x'][index],
                markersize=4, linewidth=2, zorder=5-index)
            if errors is not None:
                sd = errors[errors.method.eq(method)].set_index('requested_k').loc[curve.requested_k, metric].to_numpy()
                ax.fill_between(curve.requested_k, np.clip(curve[metric]-sd, 0, 1), np.clip(curve[metric]+sd, 0, 1), color=color, alpha=.06)
            elif 'cv_selected_k' in curve and metric == 'auroc':
                point = curve[curve.requested_k.eq(curve.cv_selected_k)]
                ax.scatter(point.requested_k, point[metric], color=color, marker='*', s=130, edgecolor='white')
        ax.set(title=metric.upper(), xlabel='Requested K', ylabel=metric.upper(), xticks=ks)
        ax.grid(alpha=.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, .13, 1, .94))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def convergence_figures(dataset):
    root = OUTPUT/dataset
    profiles, summaries = [], []
    for space in ['fold_0', 'fold_1', 'fold_2', 'outer']:
        trace = pd.read_csv(root/space/'clustering/convergence.csv.gz')
        for resolution, part in trace.groupby('resolution'):
            weights = part.groupby('component').component_weight.first()
            wide = part.pivot(index='component', columns='extra_iterations', values='quality').ffill(axis=1)
            quality = wide.mul(weights/weights.sum(), axis=0).sum()
            for extra, value in quality.items():
                rows = part[part.extra_iterations.eq(extra)]
                profiles.append(dict(space=space, resolution=resolution, extra_iterations=extra,
                    global_quality=float(value), active_components=len(rows),
                    membership_changes=int(rows.membership_changes.sum())))
            final = part.sort_values('extra_iterations').groupby('component').tail(1)
            summaries.append(dict(space=space, resolution=resolution, components=len(final),
                minimum_extra_iterations=int(final.extra_iterations.min()), maximum_extra_iterations=int(final.extra_iterations.max()),
                nonconverged_components=int((~final.converged).sum())))
    pd.DataFrame(summaries).to_csv(root/'convergence_summary.csv', index=False)
    pd.DataFrame(profiles).to_csv(root/'convergence_profiles.csv', index=False)
    # Compare paired first/final component qualities; don't sum a changing subset.
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    outer = pd.read_csv(root/'outer/clustering/convergence.csv.gz')
    for resolution in RESOLUTIONS:
        part = outer[outer.resolution.eq(resolution)]
        initial = part[part.extra_iterations.eq(0)].set_index('component').quality
        gain = (part.quality-part.component.map(initial)).clip(lower=0)
        for extra, rows in part.assign(gain=gain).groupby('extra_iterations'):
            axes[0].scatter(extra, rows.gain.max(), color=COLORS[list(RESOLUTIONS).index(resolution)], s=12)
        final = part.sort_values('extra_iterations').groupby('component').tail(1)
        axes[1].hist(final.extra_iterations, bins=20, alpha=.4, label=f'resolution {resolution:g}')
    axes[0].set(title='Maximum component gain over initial convergence', xlabel='Additional iterations', ylabel='Normalized modularity gain')
    axes[1].set(title='Additional iterations before verified convergence', xlabel='Additional iterations', ylabel='Components')
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(root/'convergence.png', dpi=180)
    plt.close(fig)


def dataset_run(dataset):
    started = time.monotonic()
    source = PROJECT/'outputs/preprocessing'/dataset
    matrix = sparse.load_npz(source/'X_valid.npz').tocsr()
    samples = pd.read_csv(source/'samples.csv')
    y = samples.y.to_numpy(int)
    with np.load(source/'edges.npz') as saved:
        pb, pg = saved['pair_bacteria'], saved['pair_gene']
    if matrix.shape != (len(y), len(pb)) or len(pb) != len(pg) or set(np.unique(y)) != {0, 1} or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Invalid sample-by-pair matrix/labels/abundances')
    if samples.sample_id.duplicated().any() or ('subject_id' in samples and samples.subject_id.duplicated().any()):
        raise ValueError('Only one independent sample per participant is allowed')
    if np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0):
        raise ValueError('All-zero sample')
    a, b = next(StratifiedShuffleSplit(1, test_size=.2, random_state=SEED).split(matrix, y))
    directory = OUTPUT/dataset
    directory.mkdir(parents=True, exist_ok=True)
    frames, summaries = [], []
    for fold, (aa, bb) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(a, y[a])):
        metrics, diagnostics = prepare_space(matrix, y, pb, pg, dataset, f'fold_{fold}', a[aa], a[bb], b)
        frames.append(metrics.assign(fold=fold))
        summaries.append(diagnostics)
    cv = pd.concat(frames, ignore_index=True)
    cv.to_csv(directory/'inner_metrics.csv', index=False)
    mean, locked = choose_settings(cv[cv.method.ne('fresh_control')])
    mean.to_csv(directory/'mean_cv_by_setting.csv', index=False)
    locked.to_csv(directory/'locked_parameters.csv', index=False)
    chosen_settings = [s for s in settings() if s['setting'] in set(locked.setting)]
    # Lock settings before reading holdout predictions or constructing outer fits.
    metrics, diagnostic = prepare_space(matrix, y, pb, pg, dataset, 'outer', a, b, b, selected=chosen_settings)
    summaries.append(diagnostic)
    control = metrics[metrics.method.eq('fresh_control')]
    control.to_csv(directory/'fresh_control_metrics.csv', index=False)
    metrics = metrics[metrics.method.ne('fresh_control')].copy()
    mapping = locked.set_index('method').requested_k
    metrics['cv_selected_k'] = metrics.method.map(mapping)
    metrics['dataset'] = dataset
    metrics.to_csv(directory/'holdout_metrics.csv', index=False)
    chosen = metrics[metrics.requested_k.eq(metrics.cv_selected_k)].merge(locked[['method', 'mean_cv_auc']], on='method')
    chosen.to_csv(directory/'cv_selected_models.csv', index=False)
    pd.concat(summaries, ignore_index=True).to_csv(directory/'cluster_diagnostics.csv', index=False)
    original = pd.read_csv(cached_root(dataset)/'metrics.csv', float_precision='round_trip')
    for method in ('bacteria', 'baseline'):
        old = original[original.method.eq(method)].sort_values('requested_k')
        new = metrics[metrics.method.eq(method)].sort_values('requested_k')
        np.testing.assert_allclose(new[['effective_k', *METRICS]], old[['effective_k', *METRICS]], rtol=0, atol=1e-12)
    overlaps = []
    for method in METHODS:
        one = metrics[metrics.method.eq(method)].sort_values('requested_k')
        for other in METHODS:
            if list(METHODS).index(other) <= list(METHODS).index(method):
                continue
            two = metrics[metrics.method.eq(other)].sort_values('requested_k')
            if np.array_equal(one[METRICS].to_numpy(), two[METRICS].to_numpy()):
                overlaps.append(dict(method=method, identical_curve_to=other))
    pd.DataFrame(overlaps, columns=['method', 'identical_curve_to']).to_csv(directory/'identical_curves.csv', index=False)
    for grid, ks in GRIDS.items():
        plot_curves(metrics[metrics.requested_k.isin(ks)],
            f'{dataset} — Bernoulli clustering ({grid} K)\nAll selection MW/BH; ★ training-CV K; holdout exploratory',
            directory/f'{grid}_k_metrics.png', ks)
    convergence_figures(dataset)
    write_json(directory/'result.json', dict(status='complete', references_reproduced=True, wall_seconds=time.monotonic()-started,
        signature=json.loads((OUTPUT/'protocol.json').read_text())['signature']))
    print(f'COMPLETE {dataset}: five curves and fresh convergence diagnostic', flush=True)


def summarize():
    complete = [d for d in DATASETS if (OUTPUT/d/'result.json').exists() and json.loads((OUTPUT/d/'result.json').read_text()).get('status') == 'complete']
    if not complete:
        return
    data = pd.concat([pd.read_csv(OUTPUT/d/'holdout_metrics.csv') for d in complete], ignore_index=True)
    data.to_csv(OUTPUT/'metrics.csv', index=False)
    chosen = pd.concat([pd.read_csv(OUTPUT/d/'cv_selected_models.csv') for d in complete], ignore_index=True)
    chosen.to_csv(OUTPUT/'cv_selected_models.csv', index=False)
    peaks = []
    for grid, ks in {**GRIDS, 'union': K_VALUES}.items():
        subset = data[data.requested_k.isin(ks)]
        best = subset.sort_values('requested_k').loc[subset.sort_values('requested_k').groupby(['dataset', 'method']).auroc.idxmax()].copy()
        refs = best[best.method.isin(('baseline', 'bacteria'))].pivot(index='dataset', columns='method', values='auroc')
        best['peak_gap_to_bacteria'] = best.dataset.map(refs.bacteria)-best.auroc
        best['peak_improvement_vs_baseline'] = best.auroc-best.dataset.map(refs.baseline)
        peaks.append(best.assign(grid=grid))
        if grid == 'union':
            continue
        grouped = subset.groupby(['method', 'requested_k'])[METRICS]
        mean, sd = grouped.mean().reset_index(), grouped.std().reset_index()
        mean.to_csv(OUTPUT/f'{grid}_k_aggregate_mean.csv', index=False)
        sd.to_csv(OUTPUT/f'{grid}_k_aggregate_sd.csv', index=False)
        plot_curves(mean, f'{len(complete)}/8 real datasets — equal-weight mean ± SD ({grid} K)\nExploratory; SD is not a confidence interval',
            OUTPUT/f'{grid}_k_aggregate_real.png', ks, sd)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
        for ax, dataset in zip(axes.flat, DATASETS):
            if dataset not in complete:
                ax.set(title=f'{dataset}: pending')
                continue
            for index, ((method, label), color) in enumerate(zip(METHODS.items(), COLORS)):
                curve = subset[subset.dataset.eq(dataset) & subset.method.eq(method)].sort_values('requested_k')
                ax.plot(curve.requested_k, curve.auroc, label=label, color=color,
                    linestyle=['-', '--', ':', '-.', '--'][index], marker=['o', 's', '^', 'D', 'x'][index], markersize=3)
            ax.set(title=dataset, xlabel='Requested K', ylabel='AUROC', xticks=ks)
            ax.grid(alpha=.2)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=9)
        fig.suptitle(f'Bernoulli clustering: {len(complete)}/8 completed datasets ({grid} K)')
        fig.tight_layout(rect=(0, .13, 1, .94))
        fig.savefig(OUTPUT/f'{grid}_k_all_real_auc.png', dpi=180)
        plt.close(fig)
    pd.concat(peaks, ignore_index=True).to_csv(OUTPUT/'peak_summary.csv', index=False)
    lines = ['# Fixed-graph Bernoulli clustering analysis', '', f'Completed: {len(complete)}/8 datasets ({", ".join(complete)}).', '',
        'All five main curves use unchanged MW/BH q≤.10 and final balanced L2 logistic C=1. Clustering parameters and operating K are chosen solely by three-fold training-validation AUROC.', '',
        'Leiden runs until stable, then at least 100 extra iterations in batches of 25, requiring three successive gains ≤1e-10; cap 1,000. Inspect convergence_summary.csv and convergence.png.', '',
        'Hierarchy merges all cached baseline communities using strongest inter-community Bernoulli edges. Average/complete linkage and similarity cut are CV-selected. Missing edges have similarity zero; disconnected components never merge.', '',
        'Fresh standard-convergence Leiden is a diagnostic, not a replacement for the cached baseline. IDs retain baseline tie order for unchanged partitions. Identical curves are listed per dataset and retain distinct plot markers.', '',
        'cv_selected_models.csv reports prediction at training-selected K. peak_summary.csv contains descriptive fine/coarse/union holdout maxima, not deployment settings. All holdouts remain exploratory.', '',
        'Sparse representations, original pair membership, selected features, cluster statistics and within-Top-10 signed/absolute correlations are saved by dataset/space/setting. Source graphs and historical comparison outputs remain unchanged.']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paths = [Path(__file__), WORKER, PROJECT/'research_utils.py', PROJECT/'scripts/run_qinj_redundancy_analysis.py',
             PROJECT/'scripts/run_method_comparison.py', PROJECT/'scripts/run_bernoulli_correlation_join.py', PROJECT/'feature_selection/run_feature_selection.py',
             JOIN_PYTHON.parents[2]/'community_detection.py', JOIN_PYTHON.parents[2]/'bernoulli_join.py']
    for dataset in DATASETS:
        paths.extend(PROJECT/'outputs/preprocessing'/dataset/name for name in ('X_valid.npz', 'samples.csv', 'edges.npz'))
        paths.append(cached_root(dataset)/'metrics.csv')
        for name in ('fold_0', 'fold_1', 'fold_2', 'outer'):
            source = cached_root(dataset)/name
            graph_root = PROJECT/'outputs/method_comparison/real'/dataset/'bernoulli_join_graph' if name == 'outer' else source/'join'
            graph, _ = graph_path(graph_root)
            paths.extend([source/'split.npz', graph_root/'assignments.npz', graph_root/'integration.json', graph/'run.json', graph/'nodes.npy',
                          *sorted((graph/'edges').glob('part-*.npy'))])
            paths.extend(source/f'{family}_{suffix}' for family in ('join', 'bacteria') for suffix in ('train.npz', 'test.npz', 'features.csv.gz'))
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    graph_versions = json.loads(subprocess.check_output([str(JOIN_PYTHON), '-c',
        'import json,sys,numpy,scipy,igraph; print(json.dumps(dict(python=sys.version,numpy=numpy.__version__,scipy=scipy.__version__,igraph=igraph.__version__)))'], text=True))
    protocol = dict(datasets=DATASETS, settings=settings(), k_values=K_VALUES, seed=SEED, leiden_seed=LEIDEN_SEED,
        model_versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        graph_versions=graph_versions,
        minimum_extra=MIN_EXTRA, maximum_extra=MAX_EXTRA, batch=BATCH, patience=PATIENCE, quality_tolerance=QUALITY_TOL,
        graph_rebuilding=False, workers=WORKERS, memory_ceiling_gib=100, final_model='balanced L2 logistic C=1',
        selection='MW/BH q<=.10 recomputed over each complete partition; no inherited BH screening',
        cv='Three independently preprocessed cached inner graphs; maximize mean validation AUROC, listed parameter order then smaller K',
        hierarchy='All baseline communities; strongest original cross edge; missing edge similarity zero; independent connected components',
        holdout='Previously inspected: exploratory')
    signature = hashlib.sha256(repr((states, protocol)).encode()).hexdigest()
    marker = OUTPUT/'protocol.json'
    if marker.exists() and json.loads(marker.read_text())['signature'] != signature:
        raise ValueError('Changed source/protocol; preserve old caches and use a new output directory')
    write_json(marker, dict(signature=signature, source_states=states, **protocol))
    status = [dict(dataset=d, status='pending') for d in DATASETS]
    for row in status:
        dataset = row['dataset']
        row['status'] = 'running'
        pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
        done = OUTPUT/dataset/'result.json'
        try:
            if done.exists() and json.loads(done.read_text()).get('status') == 'complete':
                if json.loads(done.read_text())['signature'] != signature:
                    raise ValueError('Stale completed dataset')
                print(f'Reusing completed {dataset}', flush=True)
            else:
                dataset_run(dataset)
            row['status'] = 'complete'
            summarize()
        except Exception as error:
            traceback.print_exc()
            row.update(status='failed', error=str(error))
            done.parent.mkdir(parents=True, exist_ok=True)
            write_json(done, row)
            if dataset in PILOTS:
                pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
                raise RuntimeError('Pilot validation failed; remaining datasets were not launched') from error
        pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
    unchanged = states == [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    complete = all(row['status'] == 'complete' for row in status) and unchanged
    write_json(OUTPUT/'result.json', dict(status='complete' if complete else 'failed', datasets=status, source_caches_unchanged=unchanged))
    if not complete:
        raise RuntimeError('Incomplete analysis or changed source caches; see status.csv')
    print('COMPLETE: all eight datasets; five methods; fine/coarse comparison', flush=True)


if __name__ == '__main__':
    main()
