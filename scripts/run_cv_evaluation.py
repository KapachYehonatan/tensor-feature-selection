#!/usr/bin/env python3
"""Five-fold full-cohort evaluation; every learned representation is fold-local."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
import warnings

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[name] = '1'
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from research_utils import (normalize_rows, prevalence_mask, collapsed_average,
    aggregate_pair_clusters, JOIN_PYTHON, screen, evaluate, K_VALUES, SEED, cache_signature)
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = PROJECT/'outputs/cv_evaluation'
DATASETS = ['HMP_2019_ibdmdb_unique', 'QinJ_2012']
THRESHOLDS = {'HMP_2019_ibdmdb_unique': .70, 'QinJ_2012': .80}
FOLDS = 5
JOIN_WORKERS, COMMUNITY_WORKERS = 4, 1
MEMORY_LIMIT_GIB = 200  # Hard cgroup ceiling; includes the runner and page cache.
WORKER_MEMORY_GIB = 125  # Operating budget leaves room for the runner and host.
JOIN_WORKING_GIB, COMMUNITY_WORKING_GIB = 24, 125
RESERVE_GIB, FREE_DISK_GIB, OUTPUT_GIB, KEEP_GRAPH_GIB = 16, 500, 40, 40
EDGE_LIMIT = 500_000_000
WORKER = PROJECT/'scripts/bernoulli_join_worker.py'
GRIDS = {'fine': list(range(1, 11)), 'coarse': [5, 10, 20, 35, 50, 60, 75, 85, 100]}
METRICS = ['auroc', 'precision', 'recall', 'f1']
METHODS = {'bacteria': 'Bacteria-collapsed | MW/BH', 'gene': 'Gene-collapsed | MW/BH',
           'join': 'Bernoulli-correlation-join mean | MW/BH'}
COLORS = ['#F58518', '#B279A2', '#4C78A8']


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def make_folds(target):
    target = np.asarray(target)
    if set(np.unique(target)) != {0, 1} or np.bincount(target).min() < FOLDS:
        raise ValueError('Five-fold stratification requires both classes and sufficient samples')
    folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(np.zeros(len(target)), target))
    coverage = np.zeros(len(target), int)
    for a, b in folds:
        if np.intersect1d(a, b).size or not np.array_equal(np.sort(np.r_[a, b]), np.arange(len(target))):
            raise ValueError('Invalid CV split')
        coverage[b] += 1
    if not np.all(coverage == 1): raise ValueError('Every sample must be validated exactly once')
    return folds


def training_pairs(matrix, train_index):
    # Never use edges.npz global_mask/global_support or validation feature values.
    mask, _, cutoff = prevalence_mask(matrix[train_index], .05, 5)
    ids = np.flatnonzero(mask)
    if not len(ids): raise ValueError('Training prevalence retained no pair features')
    return ids, cutoff


def verify_split(path, a, b, ids, cutoff):
    if path.exists():
        with np.load(path) as saved:
            for key, expected in [('train_index', a), ('test_index', b), ('feature_ids', ids), ('prevalence_threshold', cutoff)]:
                if not np.array_equal(saved[key], expected): raise ValueError(f'Changed cached {key}: {path}')
    else:
        np.savez_compressed(path, train_index=a, test_index=b, feature_ids=ids, prevalence_threshold=cutoff)


def cluster_fold(matrix_path, split_path, directory, threshold):
    marker = directory/'integration.json'
    if not marker.exists():
        # Preserve interrupted/failed private outputs rather than mixing partial graphs.
        manifest = directory/'graph/run.json'
        if directory.exists() and (not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'complete'):
            number = 1
            archive = directory.with_name(directory.name+f'.failed-{number}')
            while archive.exists():
                number += 1
                archive = directory.with_name(directory.name+f'.failed-{number}')
            directory.rename(archive)
            print(f'Preserved incomplete graph in {archive}', flush=True)
        command = [str(JOIN_PYTHON), '-u', str(WORKER), str(matrix_path), str(split_path), str(directory),
            '--threshold', str(threshold), '--fixed-threshold', '--workers', str(JOIN_WORKERS),
            '--community-workers', str(COMMUNITY_WORKERS), '--memory-gb', str(WORKER_MEMORY_GIB),
            '--working-memory-gb', str(JOIN_WORKING_GIB), '--community-working-memory-gb', str(COMMUNITY_WORKING_GIB),
            '--memory-reserve-gb', str(RESERVE_GIB), '--free-disk-gb', str(FREE_DISK_GIB),
            '--output-gb', str(OUTPUT_GIB), '--keep-graph-gb', str(KEEP_GRAPH_GIB),
            '--community-max-edges', str(EDGE_LIMIT)]
        print('Fold-local exact graph: '+' '.join(command), flush=True)
        with (directory.parent/'clustering.log').open('a') as log:
            subprocess.run(command, cwd=PROJECT, check=True, stdout=log, stderr=subprocess.STDOUT)
    info = json.loads(marker.read_text())
    community = info['community']
    if (info['status'] != 'complete' or not info['fixed_threshold']
            or info['requested_threshold'] != threshold or info['effective_threshold'] != threshold
            or community['resolution'] != 1. or community['seed'] != 17729
            or community.get('objective', 'modularity') != 'modularity'
            or community.get('leiden_iterations', -1) != -1 or info['edges_pruned']):
        raise ValueError('Fold clustering differs from the locked baseline algorithm')
    with np.load(directory/'assignments.npz') as saved:
        labels, ids = saved['clusters'], saved['feature_ids']
    with np.load(split_path) as split:
        if not np.array_equal(ids, split['feature_ids']): raise ValueError('Assignment feature IDs changed')
    if len(labels) != len(ids) or np.any(labels < 0) or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError('Invalid fold assignments')
    sizes = np.bincount(labels)
    if np.any(sizes == 0) or len(sizes) != info['clusters']: raise ValueError('Empty/missing clusters')
    return labels, info


def prepare_features(matrix, target, pb, pg, a, b, ids, labels, directory):
    marker = directory/'representations_complete.json'
    if marker.exists(): return json.loads(marker.read_text())
    train, validation = normalize_rows(matrix[a][:, ids]), normalize_rows(matrix[b][:, ids])
    counts = []
    for method in METHODS:
        if method == 'join':
            t, v, sizes = aggregate_pair_clusters(train, validation, labels)
            feature_ids, level = np.arange(len(sizes)), 'pair_cluster'
        else:
            entity = pb if method == 'bacteria' else pg
            t, feature_ids = collapsed_average(train, entity[ids], int(entity.max())+1)
            v, validation_ids = collapsed_average(validation, entity[ids], int(entity.max())+1)
            if not np.array_equal(feature_ids, validation_ids): raise ValueError('Marginal columns changed')
            sizes, level = np.bincount(entity[ids])[feature_ids], 'bacterium' if method == 'bacteria' else 'gene'
        total = t.shape[1]
        t, v, table = screen(t, v, target[a], feature_ids, level, sizes)
        if table.feature_id.duplicated().any() or not table.q.le(.10).all(): raise ValueError('Invalid BH ranking')
        table.to_csv(directory/f'{method}_features.csv.gz', index=False)
        sparse.save_npz(directory/f'{method}_train.npz', t[:, :100])
        sparse.save_npz(directory/f'{method}_test.npz', v[:, :100])
        counts.append(dict(method=method, tested_features=total, bh_significant_features=len(table),
            cached_top_k_features=min(len(table), 100), singleton_candidates=int(np.sum(sizes == 1))))
    info = dict(counts=counts, training_samples=len(a), validation_samples=len(b),
        training_classes=np.bincount(target[a]).tolist(), validation_classes=np.bincount(target[b]).tolist(),
        retained_pairs=len(ids), retained_bacteria=len(np.unique(pb[ids])), retained_genes=len(np.unique(pg[ids])),
        clusters=len(np.bincount(labels)), singleton_clusters=int(np.sum(np.bincount(labels) == 1)))
    write_json(marker, info)
    return info


def prediction_metrics(predictions):
    rows = []
    for (method, k), group in predictions.groupby(['method', 'requested_k']):
        if not np.isfinite(group.probability).all() or not group.probability.between(0, 1).all():
            raise ValueError('Invalid out-of-fold probabilities')
        y, probability = group.y.to_numpy(), group.probability.to_numpy()
        predicted = probability >= .5
        rows.append(dict(method=method, requested_k=k, auroc=roc_auc_score(y, probability),
            precision=precision_score(y, predicted, zero_division=0), recall=recall_score(y, predicted, zero_division=0),
            f1=f1_score(y, predicted, zero_division=0)))
    return pd.DataFrame(rows).sort_values(['method', 'requested_k']).reset_index(drop=True)


def fold_run(dataset, fold, matrix, samples, pb, pg, a, b, matrix_path, signature):
    directory = OUTPUT/dataset/f'fold_{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    ids, cutoff = training_pairs(matrix, a)
    verify_split(directory/'split.npz', a, b, ids, cutoff)
    complete = directory/'complete.json'
    if complete.exists():
        if json.loads(complete.read_text())['signature'] != signature: raise ValueError('Changed completed fold inputs')
        print(f'Reusing completed {dataset} fold {fold}', flush=True)
        return
    started = time.monotonic()
    print(f'{dataset} fold {fold}: {len(a)} train / {len(b)} validation; '
          f'classes={np.bincount(samples.y.to_numpy()[a]).tolist()}; retained pairs={len(ids):,}; cutoff={cutoff}', flush=True)
    labels, cluster_info = cluster_fold(matrix_path, directory/'split.npz', directory/'join_graph', THRESHOLDS[dataset])
    info = prepare_features(matrix, samples.y.to_numpy(int), pb, pg, a, b, ids, labels, directory)
    print(f'{dataset} fold {fold}: bacteria={info["retained_bacteria"]:,}, genes={info["retained_genes"]:,}, '
          f'clusters={info["clusters"]:,}, singletons={info["singleton_clusters"]:,}', flush=True)
    metrics, predictions = [], []
    y = samples.y.to_numpy(int)
    for method in METHODS:
        train, validation = [sparse.load_npz(directory/f'{method}_{suffix}.npz') for suffix in ('train', 'test')]
        if train.shape[0] != len(a) or validation.shape != (len(b), train.shape[1]):
            raise ValueError('Unaligned ranked representations')
        # Existing evaluate() fits its scaler only on training. Nonconvergence is a failure, not a replacement method.
        with warnings.catch_warnings():
            warnings.simplefilter('error', ConvergenceWarning)
            metric, prediction = evaluate(train, validation, y[a], y[b], K_VALUES)
        metrics.append(metric.assign(dataset=dataset, fold=fold, method=method))
        prediction['effective_k'] = prediction.requested_k.map(metric.set_index('requested_k').effective_k)
        predictions.append(prediction.assign(dataset=dataset, fold=fold, method=method,
            sample_index=b[prediction.test_position], sample_id=samples.sample_id.to_numpy()[b[prediction.test_position]]))
    metric, prediction = pd.concat(metrics, ignore_index=True), pd.concat(predictions, ignore_index=True)
    metric.to_csv(directory/'metrics.csv', index=False)
    prediction.to_csv(directory/'predictions.csv.gz', index=False)
    # Re-read saved probabilities to independently validate reported metrics.
    restored = pd.read_csv(directory/'predictions.csv.gz', float_precision='round_trip', dtype={'sample_id': str})
    recomputed = prediction_metrics(restored)
    np.testing.assert_allclose(metric.sort_values(['method', 'requested_k'])[METRICS], recomputed[METRICS], rtol=0, atol=1e-12)
    pd.DataFrame(info.pop('counts')).assign(dataset=dataset, fold=fold).to_csv(directory/'candidate_counts.csv', index=False)
    write_json(complete, dict(status='complete', signature=signature, wall_seconds=time.monotonic()-started,
        graph_edges=cluster_info['community']['deduplicated_edges'], effective_threshold=THRESHOLDS[dataset], **info))
    print(f'COMPLETE {dataset} fold {fold}: {time.monotonic()-started:.1f}s', flush=True)


def summarize_dataset(dataset, samples):
    root = OUTPUT/dataset
    metrics = pd.concat([pd.read_csv(root/f'fold_{f}/metrics.csv', float_precision='round_trip') for f in range(FOLDS)], ignore_index=True)
    predictions = pd.concat([pd.read_csv(root/f'fold_{f}/predictions.csv.gz', float_precision='round_trip', dtype={'sample_id': str})
                             for f in range(FOLDS)], ignore_index=True)
    keys = ['fold', 'method', 'requested_k']
    expected = {(f, m, k) for f in range(FOLDS) for m in METHODS for k in K_VALUES}
    if metrics.duplicated(keys).any() or set(map(tuple, metrics[keys].to_numpy())) != expected or not np.isfinite(metrics[METRICS]).all().all():
        raise ValueError('Incomplete fold/method/K grid')
    if predictions.duplicated(['method', 'requested_k', 'sample_index']).any(): raise ValueError('Duplicate OOF coverage')
    for (method, k), group in predictions.groupby(['method', 'requested_k']):
        ordered = group.sort_values('sample_index')
        if (not np.array_equal(ordered.sample_index, np.arange(len(samples)))
                or not np.array_equal(ordered.y, samples.y)
                or not np.array_equal(ordered.sample_id.astype(str), samples.sample_id.astype(str))):
            raise ValueError('Incomplete/misaligned OOF sample coverage')
    if set(predictions.method) != set(METHODS) or set(predictions.requested_k) != set(K_VALUES):
        raise ValueError('Missing OOF method/K')
    for fold, group in predictions.groupby('fold'):
        with np.load(root/f'fold_{fold}/split.npz') as split:
            validation_index = split['test_index']
        for _, point in group.groupby(['method', 'requested_k']):
            if not np.array_equal(np.sort(point.sample_index), np.sort(validation_index)):
                raise ValueError('OOF predictions include training samples or omit validation samples')
        new = prediction_metrics(group)
        old = metrics[metrics.fold.eq(fold)].sort_values(['method', 'requested_k'])
        np.testing.assert_allclose(old[METRICS], new[METRICS], rtol=0, atol=1e-12)
    grouped = metrics.groupby(['method', 'requested_k'])
    mean, sd = grouped[METRICS].mean().reset_index(), grouped[METRICS].std().reset_index()
    effective = grouped.effective_k.agg(['min', 'max', 'mean']).reset_index().rename(
        columns={'min': 'effective_k_min', 'max': 'effective_k_max', 'mean': 'effective_k_mean'})
    mean = mean.merge(effective, on=['method', 'requested_k'])
    pooled = prediction_metrics(predictions).merge(effective, on=['method', 'requested_k'])
    metrics.to_csv(root/'fold_metrics.csv', index=False)
    predictions.to_csv(root/'oof_predictions.csv.gz', index=False)
    mean.assign(n_folds=FOLDS).to_csv(root/'mean_fold_metrics.csv', index=False)
    sd.to_csv(root/'sd_fold_metrics.csv', index=False)
    pooled.to_csv(root/'pooled_oof_metrics.csv', index=False)
    pd.concat([pd.read_csv(root/f'fold_{f}/candidate_counts.csv') for f in range(FOLDS)]).to_csv(root/'candidate_counts.csv', index=False)
    for grid, ks in GRIDS.items():
        plot_metrics(mean[mean.requested_k.isin(ks)], sd[sd.requested_k.isin(ks)],
            f'{dataset}: five-fold CV ({grid} K)', root/f'{grid}_k_metrics.png', ks,
            folds=metrics[metrics.requested_k.isin(ks)])
    return mean.assign(dataset=dataset), sd.assign(dataset=dataset), pooled.assign(dataset=dataset)


def plot_metrics(mean, sd, title, path, ks, folds=None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for index, ((method, label), color) in enumerate(zip(METHODS.items(), COLORS)):
            curve = mean[mean.method.eq(method)].sort_values('requested_k')
            spread = sd[sd.method.eq(method)].set_index('requested_k').loc[curve.requested_k, metric].to_numpy()
            if folds is not None:
                for _, group in folds[folds.method.eq(method)].groupby('fold'):
                    group = group.sort_values('requested_k')
                    ax.plot(group.requested_k, group[metric], color=color, alpha=.16, linewidth=.8)
            ax.fill_between(curve.requested_k, np.clip(curve[metric]-spread, 0, 1), np.clip(curve[metric]+spread, 0, 1), color=color, alpha=.10)
            ax.plot(curve.requested_k, curve[metric], label=label, color=color,
                linestyle=['-', '--', ':'][index], marker=['o', 's', '^'][index], markersize=4, linewidth=2)
        ax.set(title=metric.upper(), xlabel='Requested K', ylabel=metric.upper(), xticks=ks)
        ax.grid(alpha=.2)
    fig.legend(*axes.flat[0].get_legend_handles_labels(), loc='lower center', fontsize=9)
    fig.suptitle(title+'\nMean ± descriptive SD (not confidence intervals); no K tuning; exploratory cohorts')
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def historical_curves():
    mapping = {'bh_bacteria': 'bacteria', 'bh_genes': 'gene', 'bernoulli_correlation_join': 'join'}
    frames = []
    for name in ('fine_k_real_metrics.csv', 'metrics_by_k.csv'):
        old = pd.read_csv(PROJECT/'outputs/method_comparison'/name, float_precision='round_trip')
        old = old[old.source.eq('real') & old.dataset.isin(DATASETS) & old.method.isin(mapping)].copy()
        old['method'] = old.method.map(mapping)
        frames.append(old[['dataset', 'method', 'requested_k', 'effective_k', *METRICS]])
    old = pd.concat(frames, ignore_index=True).drop_duplicates(['dataset', 'method', 'requested_k'])
    expected = {(d, m, k) for d in DATASETS for m in METHODS for k in K_VALUES}
    if set(map(tuple, old[['dataset', 'method', 'requested_k']].to_numpy())) != expected:
        raise ValueError('Incomplete historical reference curves')
    return old


def report(complete):
    old = historical_curves()
    old.to_csv(OUTPUT/'historical_holdout_metrics.csv', index=False)
    lines = ['# QinJ/HMP: five-fold full-pipeline evaluation', '',
        f'Completed {len(complete)}/{len(DATASETS)} datasets. See status.csv for fold failures/pending work.', '',
        'No permanent holdout and no K tuning. Seed-42 stratified folds evaluate every sample exactly once per method/K. Prevalence, exact Bernoulli graph, standard Leiden, aggregation, MW/BH and scaling are fold-local. Predeclared absolute-phi thresholds: QinJ .80, HMP .70; BH q≤.10; balanced L2 C=1.', '',
        'Fine K=1–10; coarse K={5,10,20,35,50,60,75,85,100}. Protocol fingerprints, graph/model versions and scientific settings are recorded in protocol.json.', '',
        'Primary curves average five separately computed fold metrics. Shading/SD describes variability and is NOT a confidence interval. Pooled OOF metrics use scores from different fitted models and are reported separately. Both definitions are recomputed from saved OOF probabilities. Effective K may differ between folds; exhausted pools and zero-discovery constant predictions are not excluded.', '',
        'The original holdout curves are descriptive historical references only; they do not enter training or feature selection. Previously explored cohorts make this evaluation exploratory.', '']
    if complete:
        mean = pd.concat([pd.read_csv(OUTPUT/d/'mean_fold_metrics.csv').assign(dataset=d) for d in complete], ignore_index=True)
        sd = pd.concat([pd.read_csv(OUTPUT/d/'sd_fold_metrics.csv').assign(dataset=d) for d in complete], ignore_index=True)
        pooled = pd.concat([pd.read_csv(OUTPUT/d/'pooled_oof_metrics.csv').assign(dataset=d) for d in complete], ignore_index=True)
        folds = pd.concat([pd.read_csv(OUTPUT/d/'fold_metrics.csv') for d in complete], ignore_index=True)
        for name, frame in [('mean_fold_metrics', mean), ('sd_fold_metrics', sd), ('pooled_oof_metrics', pooled), ('fold_metrics', folds)]:
            frame.to_csv(OUTPUT/f'{name}.csv', index=False)
        mean.merge(old, on=['dataset', 'method', 'requested_k'], suffixes=('_cv_mean', '_old_holdout')).to_csv(OUTPUT/'cv_vs_historical_holdout.csv', index=False)
        peaks = []
        for grid, ks in {**GRIDS, 'union': K_VALUES}.items():
            for definition, frame in [('mean_fold', mean), ('pooled_oof', pooled), ('historical_holdout', old[old.dataset.isin(complete)])]:
                part = frame[frame.requested_k.isin(ks)].sort_values('requested_k')
                best = part.loc[part.groupby(['dataset', 'method']).auroc.idxmax()].copy()
                peaks.append(best.assign(grid=grid, definition=definition))
            if grid == 'union': continue
            part = mean[mean.requested_k.isin(ks)]
            grouped = part.groupby(['method', 'requested_k'])[METRICS]
            aggregate_mean, aggregate_sd = grouped.mean().reset_index(), grouped.std().reset_index()
            aggregate_mean.assign(n_datasets=len(complete)).to_csv(OUTPUT/f'{grid}_k_aggregate_mean.csv', index=False)
            aggregate_sd.to_csv(OUTPUT/f'{grid}_k_aggregate_sd.csv', index=False)
            plot_metrics(aggregate_mean, aggregate_sd,
                f'{len(complete)}/{len(DATASETS)} datasets: equal-weight mean of CV means ({grid} K)',
                OUTPUT/f'{grid}_k_aggregate.png', ks)
        peaks = pd.concat(peaks, ignore_index=True)
        peaks.to_csv(OUTPUT/'descriptive_peaks.csv', index=False)
        lines += ['## Descriptive AUROC peaks over fine + coarse K', '',
            '| Dataset | Method | Mean-fold peak | Pooled OOF peak | Historical holdout peak |', '|---|---|---:|---:|---:|']
        wide = peaks[peaks.grid.eq('union')].pivot(index=['dataset', 'method'], columns='definition', values='auroc')
        for (dataset, method), row in wide.iterrows():
            lines.append(f'| {dataset} | {METHODS[method]} | {row.mean_fold:.3f} | {row.pooled_oof:.3f} | {row.historical_holdout:.3f} |')
        lines += ['', 'Maxima are descriptive—not performance estimates after choosing K. Aggregates weight dataset-level CV means equally; aggregate SD is between-dataset spread, distinct from within-dataset fold SD. Partial aggregates are labeled and are not two-complete-dataset results.']
    lines += ['', f'Execution: sequential folds/datasets, four exact-join workers, one Leiden worker, one native thread; hard process-tree/cgroup ceiling {MEMORY_LIMIT_GIB} GiB. The worker operating budget is {WORKER_MEMORY_GIB} GiB with {RESERVE_GIB} GiB host reserve. Graph edge ceiling remains 500 million; thresholds never change on resource failure. Logs and resource records use logs/cv_evaluation.*. Historical outputs remain untouched.']
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def build_protocol():
    paths = [Path(__file__), WORKER, PROJECT/'research_utils.py',
        JOIN_PYTHON.parents[2]/'bernoulli_join.py', JOIN_PYTHON.parents[2]/'community_detection.py']
    paths.extend(PROJECT/'outputs/preprocessing'/d/n for d in DATASETS for n in ('X_valid.npz', 'samples.csv', 'edges.npz'))
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    code = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths if p.suffix == '.py'}
    versions = json.loads(subprocess.check_output([str(JOIN_PYTHON), '-c',
        'import json,igraph,numpy,sys; print(json.dumps(dict(igraph=igraph.__version__,numpy=numpy.__version__,python=sys.version)))'], text=True))
    protocol = dict(datasets=DATASETS, folds=FOLDS, seed=SEED, k_values=K_VALUES, thresholds=THRESHOLDS,
        methods=METHODS, prevalence=[.05, 5], bh_fdr=.10, model='Training-scaled balanced L2 C=1',
        leiden=dict(resolution=1., seed=17729, iterations=-1, objective='modularity', weight='abs(phi)'),
        join_workers=JOIN_WORKERS, community_workers=COMMUNITY_WORKERS, native_threads=1,
        memory_limit_gib=MEMORY_LIMIT_GIB, worker_memory_gib=WORKER_MEMORY_GIB, edge_limit=EDGE_LIMIT,
        whole_cohort_cv=True, permanent_holdout=False, tune_k=False, reuse_historical_clusters=False,
        versions=versions, model_versions=dict(python=sys.version, numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__),
        code_hashes=code, source_states=states)
    signature = hashlib.sha256(repr(protocol).encode()).hexdigest()
    return protocol, signature, paths, states


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    protocol, signature, paths, states = build_protocol()
    manifest = OUTPUT/'protocol.json'
    signature = cache_signature(manifest, signature)
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Changed inputs/protocol; preserve existing results and use a new output directory')
    if not manifest.exists(): write_json(manifest, dict(signature=signature, **protocol))
    status = [dict(dataset=d, fold=f, status='pending') for d in DATASETS for f in range(FOLDS)]
    complete = []
    report(complete)
    for dataset in DATASETS:
        source = PROJECT/'outputs/preprocessing'/dataset
        samples = pd.read_csv(source/'samples.csv', dtype={'sample_id': str})
        y = samples.y.to_numpy()
        matrix = sparse.load_npz(source/'X_valid.npz').tocsr()
        with np.load(source/'edges.npz') as saved: pb, pg = saved['pair_bacteria'], saved['pair_gene']
        if (matrix.shape != (len(samples), len(pb)) or len(pg) != len(pb) or not np.isfinite(matrix.data).all()
                or np.any(matrix.data < 0) or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
                or samples.sample_id.isna().any() or samples.sample_id.duplicated().any()
                or ('subject_id' in samples and (samples.subject_id.isna().any() or samples.subject_id.duplicated().any()))
                or not np.issubdtype(pb.dtype, np.integer) or not np.issubdtype(pg.dtype, np.integer)
                or np.any(pb < 0) or np.any(pg < 0)):
            raise ValueError('Invalid independent-sample metadata/matrix/pair identities')
        print(f'{dataset}: all {len(samples)} samples, classes={np.bincount(y.astype(int)).tolist()}, '
              f'pairs={len(pb):,}, bacteria={len(np.unique(pb)):,}, genes={len(np.unique(pg)):,}', flush=True)
        for fold, (a, b) in enumerate(make_folds(y)):
            row = next(r for r in status if r['dataset'] == dataset and r['fold'] == fold)
            row['status'] = 'running'
            pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
            try:
                fold_run(dataset, fold, matrix, samples, pb, pg, a, b, source/'X_valid.npz', signature)
                row['status'] = 'complete'
            except Exception as error:
                traceback.print_exc()
                row.update(status='failed', error=str(error))
                print(f'FAILED {dataset} fold {fold}; preserving outputs, no threshold fallback; remaining dataset folds pending', flush=True)
                break
            finally:
                pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
        if all(r['status'] == 'complete' for r in status if r['dataset'] == dataset):
            summarize_dataset(dataset, samples)
            complete.append(dataset)
        report(complete)
        del matrix
    unchanged = states == [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    done = len(complete) == len(DATASETS) and unchanged
    write_json(OUTPUT/'result.json', dict(status='complete' if done else 'incomplete',
        completed_datasets=complete, folds=status, source_caches_unchanged=unchanged, signature=signature))
    if not done: raise RuntimeError('Incomplete CV evaluation; see fold status; no full aggregate claim')
    print('COMPLETE: two datasets, five full-pipeline folds each, three methods, all K', flush=True)


if __name__ == '__main__': main()
