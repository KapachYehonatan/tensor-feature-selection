#!/usr/bin/env python3
"""Test-free selectors on all cached Bernoulli-join means; never build graphs."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
import traceback
import warnings

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from research_utils import normalize_rows, prevalence_mask
from scripts.run_bernoulli_correlation_join import aggregate_pair_clusters
from scripts.run_qinj_redundancy_analysis import evaluate, K_VALUES, SEED
from scripts.run_posthoc_penalty_search import cached_root
from scripts.plot_aggregate_coverage_comparison import aggregate
from scripts.plot_best_current_qinj_comparison import GRIDS
from scripts.summarize_method_comparison import METRICS, data_limits
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT = PROJECT / 'outputs/test_free_feature_selection'
# HMP is the initial small real-data smoke run; the remaining datasets follow.
DATASETS = ['HMP_2019_ibdmdb_unique', 'QinJ_2012', 'JieZ_2017', 'LiJ_2017',
            'NielsenHB_2014_unique', 'RubelMA_2020', 'WirbelJ_2018', 'ZhuF_2020']
C_VALUES = [.01, .10, 1., 10.]
SHORTLIST = 2000
MI_BLOCK = 512
FIT_LIMITS = (2000, 10000)
TOL = 1e-4
SCORE_EPS = 1e-12  # Numerical zero, not a statistical significance threshold.
METHODS = {'bacteria': 'Bacteria-collapsed | BH reference',
           'baseline': 'Bernoulli-join mean | BH reference',
           'mi': 'Bernoulli-join mean | zero-aware mutual information',
           'l2': 'Bernoulli-join mean | L2 logistic importance',
           'l1': 'Bernoulli-join mean | L1 logistic importance',
           'elastic_net': 'Bernoulli-join mean | elastic-net logistic importance',
           'mi_l1': 'Bernoulli-join mean | MI top 2,000 → L1 logistic'}
GRID = {'mi': [0.], **{m: C_VALUES for m in ('l2', 'l1', 'elastic_net', 'mi_l1')}}
COLORS = ['#F58518', '#4C78A8', '#54A24B', '#B279A2', '#111111', '#E45756', '#17BECF']


def rank_scores(scores, ids, maximum=100):
    scores, ids = np.asarray(scores, float), np.asarray(ids)
    if scores.shape != ids.shape or len(np.unique(ids)) != len(ids) or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Invalid feature scores/IDs')
    order = np.lexsort((ids, -scores))
    return order[scores[order] > SCORE_EPS][:maximum]


def three_states(values):
    """Training-only discretization: zero / low positive / high positive."""
    values = np.asarray(values, float)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError('Invalid nonnegative MI inputs')
    medians = np.zeros(values.shape[1])
    for j in range(values.shape[1]):
        positive = values[:, j][values[:, j] > 0]
        if len(positive): medians[j] = np.median(positive)
    return ((values > 0).astype(np.int8) + (values > medians).astype(np.int8)), medians


def mi_scores(train, target):
    target = np.asarray(target)
    if train.shape[0] != len(target) or set(np.unique(target)) != {0, 1}:
        raise ValueError('MI requires aligned binary training labels')
    scores = np.zeros(train.shape[1])
    # ponytail: only bounded sample-by-cluster blocks are dense; never raw pairs.
    for start in range(0, train.shape[1], MI_BLOCK):
        end = min(start+MI_BLOCK, train.shape[1])
        states, _ = three_states(train[:, start:end].toarray())
        scores[start:end] = mutual_info_classif(states, target, discrete_features=True,
                                                random_state=SEED, n_jobs=1)
        if end == train.shape[1] or (start//MI_BLOCK+1) % 32 == 0:
            print(f'  MI: {end:,}/{train.shape[1]:,} clusters', flush=True)
    return scores


def logistic_scores(scaled, target, method, c):
    """Coefficient ranking only; the final predictor remains the existing L2."""
    if method not in ('l2', 'l1', 'elastic_net', 'mi_l1') or c <= 0 or not np.isfinite(c):
        raise ValueError('Invalid logistic selector')
    if scaled.shape[0] != len(target) or set(np.unique(target)) != {0, 1}:
        raise ValueError('Selector requires aligned binary training labels')
    if not scaled.shape[1]: return np.zeros(0), dict(status='complete', iterations=0, attempts=0, fit_seconds=0.)
    started = time.monotonic()
    ratio = 0. if method == 'l2' else .5 if method == 'elastic_net' else 1.
    for attempt, limit in enumerate(FIT_LIMITS, 1):
        print(f'  Fit {method} C={c:g}: {scaled.shape[1]:,} candidates, max_iter={limit}', flush=True)
        model = LogisticRegression(C=c, l1_ratio=ratio, class_weight='balanced',
            solver='saga' if method == 'elastic_net' else 'liblinear', random_state=SEED,
            max_iter=limit, tol=TOL)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            model.fit(scaled, target)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        for warning in caught:
            if not issubclass(warning.category, ConvergenceWarning): print(f'  Warning: {warning.message}', flush=True)
        if converged:
            coef = model.coef_[0]
            if not np.isfinite(coef).all(): raise ValueError('Nonfinite selector coefficients')
            return coef, dict(status='complete', iterations=int(model.n_iter_.max()), attempts=attempt,
                              fit_seconds=time.monotonic()-started, solver=model.solver)
        print(f'  Nonconverged {method} C={c:g}; '+('retrying' if attempt == 1 else 'FAILED'), flush=True)
    raise RuntimeError(f'{method} C={c:g} did not converge after limits {FIT_LIMITS}; no method substituted')


def choose_settings(cv):
    keys = ['method', 'parameter', 'requested_k']
    expected = {(m, c, k) for m, cs in {'bacteria': [0.], 'baseline': [0.], **GRID}.items()
                for c in cs for k in K_VALUES}
    grouped = cv.groupby(keys)
    if cv.duplicated(['fold', *keys]).any() or set(cv.fold) != {0, 1, 2} or set(grouped.groups) != expected or not grouped.size().eq(3).all() or not np.isfinite(cv.auroc).all():
        raise ValueError('Incomplete/nonfinite three-fold method/C/K grid')
    mean = grouped.agg(mean_cv_auc=('auroc', 'mean'), mean_effective_k=('effective_k', 'mean')).reset_index()
    locked = mean.sort_values(['method', 'mean_cv_auc', 'parameter', 'requested_k'],
        ascending=[True, False, True, True]).groupby('method').head(1).copy()
    return mean, locked


def verify_split(space, outer_train, outer_test, n_samples):
    with np.load(space/'split.npz') as s:
        a, b, ids = s['train_index'], s['test_index'], s['feature_ids']
    if any(len(np.unique(x)) != len(x) or np.any(x < 0) or np.any(x >= n_samples) for x in (a, b)) or np.intersect1d(a, b).size:
        raise ValueError('Invalid/overlapping cached sample split')
    if space.name == 'outer':
        if not np.array_equal(a, outer_train) or not np.array_equal(b, outer_test): raise ValueError('Outer split changed')
    elif np.intersect1d(np.r_[a, b], outer_test).size or set(np.r_[a, b]) != set(outer_train):
        raise ValueError('Inner-fold isolation failed')
    return a, b, ids


def verify_predictors(train, test, source):
    table = pd.read_csv(source/'join_features.csv.gz', float_precision='round_trip')
    ids = table.feature_id.to_numpy(int)
    for matrix, suffix in ((train, 'train'), (test, 'test')):
        cached = sparse.load_npz(source/f'join_{suffix}.npz')
        rebuilt = matrix[:, ids]
        if rebuilt.shape != cached.shape: raise ValueError('BH predictor shape mismatch')
        difference = rebuilt-cached
        if difference.nnz and np.abs(difference.data).max() > 1e-12:
            raise ValueError('Reconstructed BH predictors changed')


def reconstruct(matrix, y, pb, pg, dataset, name, outer_train, outer_test):
    source = cached_root(dataset)/name
    a, b, ids = verify_split(source, outer_train, outer_test, len(y))
    directory = OUTPUT/dataset/name
    directory.mkdir(parents=True, exist_ok=True)
    graph = PROJECT/'outputs/method_comparison/real'/dataset/'bernoulli_join_graph' if name == 'outer' else source/'join'
    integration = json.loads((graph/'integration.json').read_text())
    with np.load(graph/'assignments.npz') as s:
        if not np.array_equal(s['feature_ids'], ids): raise ValueError('Assignment/prevalence feature IDs differ')
        labels = s['clusters']
    if integration['status'] != 'complete' or len(labels) != len(ids) or labels.min() < 0:
        raise ValueError('Invalid cached clustering')
    paths = [directory/f'all_join_{suffix}.npz' for suffix in ('train', 'test')]
    complete = directory/'representations_complete.json'
    if complete.exists():
        train, test = [sparse.load_npz(p) for p in paths]
        counts = json.loads(complete.read_text())
        if train.shape != (len(a), integration['clusters']) or test.shape != (len(b), integration['clusters']):
            raise ValueError('Unaligned all-cluster cache')
        return train, test, np.bincount(labels), a, b, counts
    mask, _, cutoff = prevalence_mask(matrix[a], .05, 5)
    if not np.array_equal(np.flatnonzero(mask), ids): raise ValueError('Training-only prevalence mask changed')
    normalized_train = normalize_rows(matrix[a][:, ids])
    normalized_test = normalize_rows(matrix[b][:, ids])
    train, test, sizes = aggregate_pair_clusters(normalized_train, normalized_test, labels)
    if len(sizes) != integration['clusters'] or np.any(sizes <= 0): raise ValueError('Missing/empty cached clusters')
    verify_predictors(train, test, source)
    counts = dict(training_samples=len(a), validation_samples=len(b), training_classes=np.bincount(y[a]).tolist(),
        retained_pairs=len(ids), retained_bacteria=len(np.unique(pb[ids])), retained_genes=len(np.unique(pg[ids])),
        clusters=len(sizes), bh_screened_clusters=len(pd.read_csv(source/'join_features.csv.gz')),
        prevalence_cutoff=int(cutoff), bernoulli_threshold=integration['effective_threshold'], bh_predictors_verified=True)
    for p, x in zip(paths, (train, test)): sparse.save_npz(p, x)
    complete.write_text(json.dumps(counts, indent=2))
    print(f'{dataset} {name}: '+json.dumps(counts), flush=True)
    return train, test, sizes, a, b, counts


def fit_rankings(train, target, sizes, directory, settings):
    scaler = StandardScaler(with_mean=False).fit(train)
    ids = np.flatnonzero(scaler.var_ > 1e-28)
    scaled = scaler.transform(train[:, ids]) if len(ids) == train.shape[1] else StandardScaler(with_mean=False).fit_transform(train[:, ids]) if len(ids) else train[:, :0]
    # The scaler above is fitted on training only; validation is never a selector input.
    mi_file = directory/'mi_scores.npz'
    if mi_file.exists():
        with np.load(mi_file) as saved:
            if not np.array_equal(saved['feature_ids'], ids): raise ValueError('MI feature IDs changed')
            mi = saved['scores']
    else:
        mi = mi_scores(train[:, ids], target)
        np.savez_compressed(mi_file, feature_ids=ids, scores=mi)
    results, info = {}, []
    for method, parameters in settings.items():
        for c in parameters:
            tag = f'{method}_C{c:g}'
            ranking_file, meta_file = directory/f'{tag}_selected.csv', directory/f'{tag}_fit.json'
            if meta_file.exists() and json.loads(meta_file.read_text())['status'] == 'complete':
                ranking = pd.read_csv(ranking_file, float_precision='round_trip')
                meta = json.loads(meta_file.read_text())
            else:
                candidate = rank_scores(mi, ids, SHORTLIST) if method == 'mi_l1' else np.arange(len(ids))
                try:
                    if method == 'mi':
                        score, coef = mi, np.full(len(ids), np.nan)
                        meta = dict(status='complete', attempts=0, iterations=0, fit_seconds=0.)
                    else:
                        coef, meta = logistic_scores(scaled[:, candidate], target, method, c)
                        score = np.abs(coef)
                    picked = rank_scores(score, ids[candidate])
                    feature_ids = ids[candidate[picked]]
                    ranking = pd.DataFrame(dict(selection_rank=np.arange(1, len(picked)+1), feature_id=feature_ids,
                        score=score[picked], coefficient=coef[picked], pair_count=sizes[feature_ids]))
                    meta.update(method=method, parameter=c, all_clusters=train.shape[1], nonconstant_clusters=len(ids),
                        candidates=len(candidate), positive_scores=int(np.sum(score > SCORE_EPS)), selected=len(picked))
                    ranking.to_csv(ranking_file, index=False)
                    meta_file.write_text(json.dumps(meta, indent=2))
                except Exception as error:
                    meta_file.write_text(json.dumps(dict(status='failed', method=method, parameter=c, error=str(error)), indent=2))
                    raise
            if ranking.feature_id.duplicated().any() or not ranking.feature_id.isin(ids).all() or not ranking.selection_rank.eq(np.arange(1, len(ranking)+1)).all():
                raise ValueError('Invalid cached feature ranking')
            results[(method, c)] = ranking.feature_id.to_numpy(int)
            info.append(meta)
            print(f'  {method} C={c:g}: positive scores={meta["positive_scores"]:,}, Top-K prefix={len(ranking)}', flush=True)
    pd.DataFrame(info).to_csv(directory/'selector_counts.csv', index=False)
    return results


def reference_cv(dataset):
    source = PROJECT/'outputs/posthoc_penalty_search/coverage_partition_transfer'/dataset/'inner_metrics.csv'
    cv = pd.read_csv(source, float_precision='round_trip')
    cv = cv[cv.family.isin(['baseline', 'bacteria'])].rename(columns={'family': 'method', 'strength': 'parameter'})
    return cv


def dataset_run(dataset):
    root = cached_root(dataset)
    done = json.loads((root/'complete.json').read_text())
    if done.get('status') != 'complete' or done['signature'] != json.loads((root/'protocol.json').read_text())['signature']:
        raise ValueError('Incomplete original representation experiment')
    source = PROJECT/'outputs/preprocessing'/dataset
    matrix = sparse.load_npz(source/'X_valid.npz').tocsr()
    samples = pd.read_csv(source/'samples.csv')
    y = samples.y.to_numpy(int)
    with np.load(source/'edges.npz') as saved: pb, pg = saved['pair_bacteria'], saved['pair_gene']
    if matrix.shape != (len(y), len(pb)) or len(pg) != len(pb) or set(np.unique(y)) != {0, 1} or not np.isfinite(matrix.data).all() or np.any(matrix.data < 0):
        raise ValueError('Invalid matrix orientation, labels or abundances')
    if samples.sample_id.duplicated().any() or ('subject_id' in samples and samples.subject_id.duplicated().any()):
        raise ValueError('Only independent participants with unique sample IDs are in scope')
    outer_train, outer_test = next(StratifiedShuffleSplit(1, test_size=.2, random_state=SEED).split(matrix, y))
    print(f'{dataset}: samples={len(y)}, classes={np.bincount(y).tolist()}, pairs={len(pb):,}, bacteria={len(np.unique(pb))}, genes={len(np.unique(pg))}', flush=True)
    directory = OUTPUT/dataset
    directory.mkdir(parents=True, exist_ok=True)
    frames, counts = [reference_cv(dataset)], []
    for fold, (aa, bb) in enumerate(StratifiedKFold(3, shuffle=True, random_state=SEED).split(outer_train, y[outer_train])):
        name = f'fold_{fold}'
        train, validation, sizes, a, b, counter = reconstruct(matrix, y, pb, pg, dataset, name, outer_train, outer_test)
        if not np.array_equal(a, outer_train[aa]) or not np.array_equal(b, outer_train[bb]): raise ValueError('Cached fold does not match deterministic CV')
        counts.append(dict(space=name, **counter))
        rankings = fit_rankings(train, y[a], sizes, directory/name, GRID)
        # Reproduce both cached reference curves without performing statistical tests.
        for family in ('baseline', 'bacteria'):
            t = sparse.load_npz(root/name/f'{"join" if family == "baseline" else family}_train.npz')[:, :100]
            v = sparse.load_npz(root/name/f'{"join" if family == "baseline" else family}_test.npz')[:, :100]
            metric, _ = evaluate(t, v, y[a], y[b], K_VALUES)
            old = frames[0][frames[0].fold.eq(fold) & frames[0].method.eq(family)].sort_values('requested_k')
            np.testing.assert_allclose(metric[['effective_k', *METRICS]], old[['effective_k', *METRICS]], rtol=0, atol=1e-12)
        for (method, c), ids in rankings.items():
            path = directory/name/f'{method}_C{c:g}_metrics.csv'
            if path.exists(): metric = pd.read_csv(path, float_precision='round_trip')
            else:
                metric, _ = evaluate(train[:, ids], validation[:, ids], y[a], y[b], K_VALUES)
                metric.to_csv(path, index=False)
            frames.append(metric.assign(fold=fold, method=method, parameter=c))
        del train, validation
        print(f'{dataset} {name}: training/validation metrics and reference reproduction complete', flush=True)
    cv = pd.concat(frames, ignore_index=True)
    mean, locked = choose_settings(cv)
    cv.to_csv(directory/'inner_metrics.csv', index=False)
    mean.to_csv(directory/'mean_cv_by_setting.csv', index=False)
    locked.to_csv(directory/'locked_parameters.csv', index=False)
    # Parameter and operating K are locked before fitting/accessing holdout predictions.
    train, test, sizes, a, b, counter = reconstruct(matrix, y, pb, pg, dataset, 'outer', outer_train, outer_test)
    counts.append(dict(space='outer', **counter))
    rankings = fit_rankings(train, y[a], sizes, directory/'outer',
                            {row.method: [float(row.parameter)] for row in locked.itertuples() if row.method in GRID})
    metric_frames, predictions, chosen = [], [], []
    original = pd.read_csv(root/'metrics.csv', float_precision='round_trip')
    for setting in locked.itertuples():
        method, c, k = setting.method, float(setting.parameter), int(setting.requested_k)
        if method in GRID:
            ids = rankings[(method, c)]
            t, v = train[:, ids], test[:, ids]
        else:
            family = 'join' if method == 'baseline' else 'bacteria'
            t = sparse.load_npz(root/'outer'/f'{family}_train.npz')[:, :100]
            v = sparse.load_npz(root/'outer'/f'{family}_test.npz')[:, :100]
        metric, prediction = evaluate(t, v, y[a], y[b], K_VALUES)
        if method not in GRID:
            old = original[original.method.eq(method)].sort_values('requested_k')
            np.testing.assert_allclose(metric[['effective_k', *METRICS]], old[['effective_k', *METRICS]], rtol=0, atol=1e-12)
        metric_frames.append(metric.assign(dataset=dataset, method=method, parameter=c, cv_selected_k=k))
        predictions.append(prediction.assign(dataset=dataset, method=method, sample_index=b[prediction.test_position]))
        point = metric[metric.requested_k.eq(k)].iloc[0].to_dict()
        chosen.append(dict(dataset=dataset, method=method, parameter=c, mean_cv_auc=float(setting.mean_cv_auc), **point))
    metrics = pd.concat(metric_frames, ignore_index=True)
    metrics.to_csv(directory/'holdout_metrics.csv', index=False)
    pd.concat(predictions, ignore_index=True).to_csv(directory/'holdout_predictions.csv.gz', index=False)
    pd.DataFrame(chosen).to_csv(directory/'cv_selected_models.csv', index=False)
    pd.DataFrame(counts).to_csv(directory/'candidate_counts.csv', index=False)
    plot_dataset(metrics, dataset)
    (directory/'result.json').write_text(json.dumps(dict(status='complete', references_reproduced=True, graph_recomputation=False), indent=2))
    print(f'COMPLETE {dataset}: all seven methods, training-only choices and fine/coarse PNGs', flush=True)


def plot_metrics(data, title, path, ks, errors=None):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for ax, metric in zip(axes.flat, METRICS):
        for (method, label), color in zip(METHODS.items(), COLORS):
            curve = data[data.method.eq(method)].sort_values('requested_k')
            ax.plot(curve.requested_k, curve[metric], color=color, label=label, marker='o', markersize=4,
                    linewidth=2, linestyle='-' if method in ('baseline', 'bacteria') else '--')
            if errors is not None:
                sd = errors[errors.method.eq(method)].set_index('requested_k').loc[curve.requested_k, metric].to_numpy()
                ax.fill_between(curve.requested_k, np.clip(curve[metric]-sd, 0, 1), np.clip(curve[metric]+sd, 0, 1), color=color, alpha=.06)
            elif metric == 'auroc':
                selected = curve[curve.requested_k.eq(curve.cv_selected_k)]
                ax.scatter(selected.requested_k, selected[metric], marker='*', color=color, s=150, edgecolor='white', zorder=5)
        ax.set(title=metric.upper(), xlabel='Requested K', ylabel=metric.upper(), xticks=ks, ylim=data_limits(data[metric]))
        ax.grid(alpha=.2)
    handles, names = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='lower center', ncol=2, fontsize=9)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=(0, .16, 1, .92))
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_dataset(metrics, dataset):
    for domain, ks in GRIDS.items():
        subset = metrics[metrics.requested_k.isin(ks)]
        plot_metrics(subset, f'{dataset} — test-free feature selection ({domain} K)\nUnchanged Bernoulli clustering and final L2 model; ★ = training-CV K; exploratory holdout',
                     OUTPUT/dataset/f'{domain}_k_metrics.png', ks)


def summarize():
    data = pd.concat([pd.read_csv(OUTPUT/d/'holdout_metrics.csv', float_precision='round_trip') for d in DATASETS], ignore_index=True)
    aggregate(data, DATASETS, K_VALUES, methods=METHODS)
    data.to_csv(OUTPUT/'metrics.csv', index=False)
    pd.concat([pd.read_csv(OUTPUT/d/'candidate_counts.csv').assign(dataset=d) for d in DATASETS], ignore_index=True).to_csv(OUTPUT/'candidate_counts.csv', index=False)
    pd.concat([pd.read_csv(OUTPUT/d/'cv_selected_models.csv') for d in DATASETS], ignore_index=True).to_csv(OUTPUT/'cv_selected_models.csv', index=False)
    peaks = []
    for domain, ks in {**GRIDS, 'fine_plus_coarse': K_VALUES}.items():
        subset = data[data.requested_k.isin(ks)]
        summary = subset.loc[subset.groupby(['dataset', 'method']).auroc.idxmax()].copy()
        refs = summary[summary.method.isin(['baseline', 'bacteria'])].pivot(index='dataset', columns='method', values='auroc')
        summary['peak_delta_vs_baseline'] = summary.auroc-summary.dataset.map(refs.baseline)
        summary['peak_delta_vs_bacteria'] = summary.auroc-summary.dataset.map(refs.bacteria)
        peaks.append(summary.assign(grid=domain))
        if domain == 'fine_plus_coarse': continue
        mean, sd = aggregate(subset, DATASETS, ks, methods=METHODS)
        subset.to_csv(OUTPUT/f'{domain}_k_metrics.csv', index=False)
        mean.to_csv(OUTPUT/f'{domain}_k_aggregate_mean.csv', index=False)
        sd.to_csv(OUTPUT/f'{domain}_k_aggregate_sd.csv', index=False)
        plot_metrics(mean, f'All 8 real datasets — test-free feature selection ({domain} K)\nEqual-weight mean ± between-dataset SD; SD is not a confidence interval; exploratory holdout',
                     OUTPUT/f'{domain}_k_aggregate_real.png', ks, sd)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True)
        for ax, dataset in zip(axes.flat, DATASETS):
            for (method, label), color in zip(METHODS.items(), COLORS):
                curve = subset[subset.dataset.eq(dataset) & subset.method.eq(method)].sort_values('requested_k')
                ax.plot(curve.requested_k, curve.auroc, label=label, color=color, linewidth=2)
            ax.set(title=dataset, xlabel='Requested K', ylabel='AUROC', xticks=ks)
            ax.tick_params(axis='x', labelsize=8)
            ax.grid(alpha=.2)
        handles, names = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, names, loc='lower center', ncol=2, fontsize=9)
        fig.suptitle(f'Test-free selection — all eight datasets ({domain} K)\nTraining-only C/K choices; exploratory holdout')
        fig.tight_layout(rect=(0, .16, 1, .92))
        fig.savefig(OUTPUT/f'{domain}_k_all_real_auc.png', dpi=180)
        plt.close(fig)
    peaks = pd.concat(peaks, ignore_index=True)
    peaks.to_csv(OUTPUT/'peak_summary.csv', index=False)
    table = peaks[peaks.grid.eq('fine_plus_coarse')].pivot(index='dataset', columns='method', values='auroc')
    lines = ['# Test-free feature selection', '', 'All eight datasets completed; both BH references reproduced.', '',
        'Five selectors use no statistical tests, bacterial predictors or BH fallback. MI uses zero/low-positive/high-positive training states; only MI → L1 uses a 2,000-feature MI shortlist.', '',
        'Selector C and operating K are selected separately for each method using mean three-fold training-validation AUROC; final prediction always uses balanced L2 C=1. Zero scores/coefficients are excluded; inspect effective K.', '',
        '## Descriptive best holdout AUROC: union of fine and coarse grids', '',
        '| Dataset | '+' | '.join(table.columns)+' |', '|---|'+'---:|'*len(table.columns)]
    lines.extend('| '+d+' | '+' | '.join(f'{v:.3f}' for v in row)+' |' for d, row in table.iterrows())
    lines.extend(['', 'These maxima are not deployment choices. `cv_selected_models.csv` reports holdout prediction at training-selected K, while `peak_summary.csv` reports separate fine/coarse/union peaks.', '',
        'Aggregate plots weight datasets equally; shading is between-dataset SD, not a confidence interval. Previously investigated holdouts make all outcomes exploratory, not independent statistical confirmation.', '',
        'Clustering was never rebuilt; original pair-count-weighted cluster means and square-root transformation were reproduced from cached assignments. Full candidate counts, selected cluster IDs, convergence records, CV profiles and sparse representation caches are saved for reproducibility.'])
    (OUTPUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    paths = [Path(__file__), PROJECT/'research_utils.py', PROJECT/'feature_selection/run_feature_selection.py',
        PROJECT/'scripts/run_bernoulli_correlation_join.py', PROJECT/'scripts/run_qinj_redundancy_analysis.py']
    for dataset in DATASETS:
        root = cached_root(dataset)
        paths.extend(PROJECT/'outputs/preprocessing'/dataset/name for name in ('X_valid.npz', 'edges.npz', 'samples.csv'))
        paths.extend([root/'complete.json', root/'protocol.json', root/'metrics.csv',
                      PROJECT/'outputs/posthoc_penalty_search/coverage_partition_transfer'/dataset/'inner_metrics.csv'])
        for name in ('fold_0', 'fold_1', 'fold_2', 'outer'):
            space = root/name
            graph = PROJECT/'outputs/method_comparison/real'/dataset/'bernoulli_join_graph' if name == 'outer' else space/'join'
            paths.extend([space/'split.npz', space/'representations_complete.json', graph/'assignments.npz', graph/'integration.json'])
            paths.extend(space/f'{family}_{suffix}' for family in ('join', 'bacteria') for suffix in ('train.npz', 'test.npz', 'features.csv.gz'))
    states = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    settings = dict(methods=METHODS, grid=GRID, k_values=K_VALUES, datasets=DATASETS, shortlist=SHORTLIST,
                    seed=SEED, fit_limits=FIT_LIMITS, tolerance=TOL, numerical_zero=SCORE_EPS)
    signature = hashlib.sha256(repr((states, settings)).encode()).hexdigest()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = OUTPUT/'protocol.json'
    if manifest.exists() and json.loads(manifest.read_text())['signature'] != signature:
        raise ValueError('Changed analysis/data inputs: preserve caches and use a new output directory')
    manifest.write_text(json.dumps(dict(signature=signature, source_states=states, **settings,
        graph_recomputation=False, candidate_pool='All training-nonconstant clusters; MI top 2,000 only in explicitly named MI→L1',
        mi='Three training-only states: zero; positive <= positive median; positive > positive median. Discrete MI, no p-values.',
        standardization='Training-only StandardScaler(with_mean=False); constant variance cutoff 1e-28',
        selection='Per-method highest mean three-fold CV AUROC over selector C/K; tie smaller C then smaller K. No baseline fallback.',
        final_model='Unchanged class-balanced L2 logistic C=1; continuous sqrt mean abundances',
        holdout='Previously investigated: exploratory, no independent validation'), indent=2))
    status = []
    for dataset in DATASETS:
        row = dict(dataset=dataset, status='running')
        status.append(row)
        pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
        marker = OUTPUT/dataset/'result.json'
        try:
            if marker.exists() and json.loads(marker.read_text())['status'] == 'complete':
                print(f'Reusing completed {dataset}', flush=True)
            else: dataset_run(dataset)
            row['status'] = 'complete'
        except Exception as error:
            traceback.print_exc()
            row.update(status='failed', error=str(error))
            (OUTPUT/dataset).mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps(row, indent=2))
            print(f'FAILED {dataset}: {error}; preserving caches, continuing other datasets', flush=True)
        pd.DataFrame(status).to_csv(OUTPUT/'status.csv', index=False)
    unchanged = states == [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    complete = all(row['status'] == 'complete' for row in status) and unchanged
    if complete: summarize()
    (OUTPUT/'result.json').write_text(json.dumps(dict(status='complete' if complete else 'failed',
        source_caches_unchanged=unchanged, graph_recomputation=False, datasets=status), indent=2))
    if not complete: raise RuntimeError('Incomplete/changed-source comparison; full aggregate withheld; see status.csv')
    print('COMPLETE: eight datasets, seven methods, fine/coarse PNGs and exploratory report', flush=True)


if __name__ == '__main__': main()
