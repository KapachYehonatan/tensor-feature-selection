"""Leakage-free five-fold prediction; all seven methods, no K tuning."""
from pathlib import Path
import json
import time
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import StratifiedKFold
from experiments.cache import (PROJECT, SOURCE, CV as OUTPUT, NATIVE_CV, PROVENANCE,
    DATASETS, K_VALUES, METRICS, SEED, NATIVE, open_protocol, complete, seal, write_json)
from methods.preprocessing import retained_pair_values, prevalence_mask, validate_samples
from methods.representations import collapsed_average, aggregate_pair_clusters
from methods.selectors.mann_whitney import mw_ranked_family
from methods.selectors import shoe
from methods.graphs.bernoulli_join import cluster
from metrics.prediction import evaluate, prediction_metrics, point_metrics
from experiments.full_dataset import prepared, select, raw_selections

FOLDS, ALPHA = 5, .10
METHODS = {'bacteria':'Bacteria-collapsed','gene':'Gene-collapsed','join':'Bernoulli-join mean'}


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


def prepare_features(matrix, target, pb, pg, a, b, ids, labels, directory):
    marker = directory/'representations_complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved.get('preprocessing') != 'supplied_values_v1': raise ValueError('Changed representation preprocessing')
        for name, size, mtime in saved['cache_states']:
            path = directory/name
            if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (size, mtime):
                raise ValueError(f'Changed cached representation: {path}')
        return saved
    train, validation = retained_pair_values(matrix[a], ids), retained_pair_values(matrix[b], ids)
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
        columns, table = mw_ranked_family(t, target[a], feature_ids, level, sizes, ALPHA)
        if table.feature_id.duplicated().any(): raise ValueError('Duplicate feature IDs')
        table.to_csv(directory/f'{method}_features.csv.gz', index=False)
        # BH/BY use the complete eligible family; prediction is unscreened.
        sparse.save_npz(directory/f'{method}_train.npz', t[:, columns].tocsr())
        sparse.save_npz(directory/f'{method}_test.npz', v[:, columns[:max(K_VALUES)]].tocsr())
        counts.append(dict(method=method, aggregated_features=total, tested_features=len(table),
            excluded_constants=total-len(table), bh_significant_features=int(table.significant_bh.sum()),
            by_significant_features=int(table.significant_by.sum()),
            cached_top_k_features=min(len(table), max(K_VALUES)), singleton_candidates=int(np.sum(sizes == 1))))
    info = dict(preprocessing='supplied_values_v1', counts=counts, training_samples=len(a), validation_samples=len(b),
        training_classes=np.bincount(target[a]).tolist(), validation_classes=np.bincount(target[b]).tolist(),
        retained_pairs=len(ids), retained_bacteria=len(np.unique(pb[ids])), retained_genes=len(np.unique(pg[ids])),
        clusters=len(np.bincount(labels)), singleton_clusters=int(np.sum(np.bincount(labels) == 1)))
    files = [directory/f'{method}_{name}' for method in METHODS
             for name in ('features.csv.gz', 'train.npz', 'test.npz')]
    info['cache_states'] = [(p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    write_json(marker, info)
    return info



def prediction_phase(dataset, fold, raw, samples, pb, pg, a, b, signature):
    directory = NATIVE_CV/dataset/f'fold_{fold}'
    directory.mkdir(parents=True, exist_ok=True)
    if complete(directory/'complete.json', signature): return
    y = samples.y.to_numpy(int)
    fit, evidence = shoe.honest_split(a, f'{dataset}/fold_{fold}')
    p = prepared(directory/'shoe', raw, fit, evidence, signature)
    selected = select(directory/'shoe', p, y, pb, pg, signature)
    evaluate_selection(directory/'shoe', p, selected, 'shoe', dataset, fold, samples, a, b, signature)
    del p
    p = prepared(directory/'raw_pairs', raw, a, None, signature)
    selections = raw_selections(directory/'raw_pairs', p, y[None], pb, pg, signature)
    for method, selected in selections.items():
        evaluate_selection(directory/method, p, selected, method, dataset, fold, samples, a, b, signature)
    for name in ['metrics.csv', 'predictions.csv.gz']:
        pd.concat([pd.read_csv(directory/m/name, dtype={'sample_id':str}) for m in NATIVE], ignore_index=True).to_csv(directory/name, index=False)
    np.savez_compressed(directory/'outer_split.npz', train_index=a, test_index=b)
    seal(directory/'complete.json', signature, [f for f in directory.rglob('*') if f.is_file() and f != directory/'complete.json'])


def analyze_method(directory, method, ytrain, ytest, signature):
    marker = directory/f'{method}_complete.json'
    if complete(marker, signature): return
    train, test = [sparse.load_npz(directory/f'{method}_{s}.npz').tocsr() for s in ('train', 'test')]
    table = pd.read_csv(directory/f'{method}_features.csv.gz', float_precision='round_trip')
    if train.shape != (len(ytrain), len(table)) or test.shape != (len(ytest), min(max(K_VALUES),len(table))):
        raise ValueError('Unaligned complete/ranked representations')
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        metrics, predictions = evaluate(train, test, ytrain, ytest, K_VALUES)
    metrics.to_csv(directory/f'{method}_metrics.csv', index=False)
    predictions['effective_k'] = predictions.requested_k.map(metrics.set_index('requested_k').effective_k)
    predictions.to_csv(directory/f'{method}_predictions.csv.gz', index=False)
    selected = [dict(selection='prediction', requested_k=k, **table.iloc[j].to_dict())
                for k in K_VALUES for j in range(min(k,len(table)))]
    pd.DataFrame(selected, columns=['selection','requested_k',*table.columns]).to_csv(
        directory/f'{method}_selected_features.csv.gz',index=False)
    seal(marker,signature,[directory/f'{method}_{n}' for n in
                           ('metrics.csv','predictions.csv.gz','selected_features.csv.gz')])


def fold_run(dataset, fold, matrix, samples, pb, pg, a, b, matrix_path, signature):
    directory = OUTPUT/dataset/f'fold_{fold}'
    directory.mkdir(parents=True,exist_ok=True)
    if complete(directory/'complete.json',signature): return
    identity=directory/'fold_protocol.json'
    if identity.exists() and json.loads(identity.read_text())['signature'] != signature:
        raise ValueError('Changed fold protocol')
    write_json(identity,dict(signature=signature,dataset=dataset,fold=fold))
    ids,cutoff=training_pairs(matrix,a)
    verify_split(directory/'split.npz',a,b,ids,cutoff)
    labels,graph=cluster(matrix_path,directory/'split.npz',directory/'join_graph')
    info=prepare_features(matrix,samples.y.to_numpy(int),pb,pg,a,b,ids,labels,directory)
    write_json(directory/'graph_summary.json',dict(dataset=dataset,fold=fold,
        requested_threshold=.70,effective_threshold=graph['effective_threshold'],
        raw_edges=graph['join']['edge_count'],graph_edges=graph['community']['deduplicated_edges'],
        threshold_edge_counts=graph['threshold_edge_counts'],resource_retries=graph.get('resource_retries',[])))
    metrics,predictions=[],[]
    y=samples.y.to_numpy(int)
    for method in METHODS:
        analyze_method(directory,method,y[a],y[b],signature)
        metrics.append(pd.read_csv(directory/f'{method}_metrics.csv',float_precision='round_trip').assign(dataset=dataset,fold=fold,method=method))
        p=pd.read_csv(directory/f'{method}_predictions.csv.gz',float_precision='round_trip')
        predictions.append(p.assign(dataset=dataset,fold=fold,method=method,sample_index=b[p.test_position],
                                    sample_id=samples.sample_id.to_numpy()[b[p.test_position]]))
    pd.concat(metrics,ignore_index=True).to_csv(directory/'metrics.csv',index=False)
    pd.concat(predictions,ignore_index=True).to_csv(directory/'predictions.csv.gz',index=False)
    pd.DataFrame(info['counts']).assign(dataset=dataset,fold=fold).to_csv(directory/'candidate_counts.csv',index=False)
    seal(directory/'complete.json',signature,[p for p in directory.rglob('*') if p.is_file() and p.name!='complete.json'])


def evaluate_selection(directory,p,selected,method,dataset,fold,samples,a,b,signature):
    directory.mkdir(parents=True,exist_ok=True)
    if complete(directory/'prediction_complete.json',signature): return
    y=samples.y.to_numpy(int)
    order=selected['order']; ranked=p['features'][:,order[:max(K_VALUES)]]
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        metrics,predictions=evaluate(ranked[a],ranked[b],y[a],y[b],K_VALUES)
    metrics.assign(dataset=dataset,fold=fold,method=method,native_discoveries=len(selected['native'])).to_csv(directory/'metrics.csv',index=False)
    predictions.assign(sample_index=b[predictions.test_position],
        sample_id=samples.sample_id.to_numpy()[b[predictions.test_position]],dataset=dataset,fold=fold,method=method).to_csv(directory/'predictions.csv.gz',index=False)
    ids=p['feature_ids']
    rows=[dict(selection='ranking',requested_k=k,rank=r+1,pair_id=int(ids[j]))
          for k in K_VALUES for r,j in enumerate(order[:k])]
    pd.DataFrame(rows,columns=['selection','requested_k','rank','pair_id']).to_csv(directory/'selected_features.csv.gz',index=False)
    seal(directory/'prediction_complete.json',signature,[directory/n for n in
        ('metrics.csv','predictions.csv.gz','selected_features.csv.gz')])


def summarize_dataset(dataset,samples,root,methods):
    directory=root/dataset
    metrics=pd.concat([pd.read_csv(directory/f'fold_{f}/metrics.csv',float_precision='round_trip') for f in range(FOLDS)],ignore_index=True)
    predictions=pd.concat([pd.read_csv(directory/f'fold_{f}/predictions.csv.gz',float_precision='round_trip',dtype={'sample_id':str}) for f in range(FOLDS)],ignore_index=True)
    expected={(f,m,k) for f in range(FOLDS) for m in methods for k in K_VALUES}
    keys=['fold','method','requested_k']
    if metrics.duplicated(keys).any() or set(map(tuple,metrics[keys].to_numpy())) != expected:
        raise ValueError('Incomplete fold/method/K grid')
    if not np.isfinite(metrics[METRICS]).all().all(): raise ValueError('Nonfinite prediction metric')
    if predictions.duplicated(['method','requested_k','sample_index']).any(): raise ValueError('Duplicate OOF prediction')
    if set(map(tuple,predictions[['method','requested_k']].drop_duplicates().to_numpy())) != {(m,k) for m in methods for k in K_VALUES}:
        raise ValueError('Incomplete prediction grid')
    for _,group in predictions.groupby(['method','requested_k']):
        ordered=group.sort_values('sample_index')
        if (not np.array_equal(ordered.sample_index,np.arange(len(samples))) or
                not np.array_equal(ordered.sample_id.astype(str),samples.sample_id.astype(str)) or
                not np.array_equal(ordered.y,samples.y)):
            raise ValueError('Incomplete/misaligned OOF coverage')
    for fold,group in predictions.groupby('fold'):
        b=np.load(OUTPUT/dataset/f'fold_{fold}/split.npz')['test_index']
        for _,point in group.groupby(['method','requested_k']):
            if not np.array_equal(np.sort(point.sample_index),np.sort(b)): raise ValueError('Validation coverage changed')
        actual=metrics[metrics.fold.eq(fold)].sort_values(['method','requested_k'])
        np.testing.assert_allclose(actual[METRICS],prediction_metrics(group)[METRICS],atol=1e-12,rtol=0)
    grouped=metrics.groupby(['method','requested_k'])
    means=grouped[METRICS].mean().reset_index()
    effective=grouped.effective_k.agg(['min','max','mean']).reset_index().rename(columns={'min':'effective_k_min','max':'effective_k_max','mean':'effective_k_mean'})
    metrics.to_csv(directory/'fold_metrics.csv',index=False)
    predictions.to_csv(directory/'oof_predictions.csv.gz',index=False)
    means.merge(effective,on=['method','requested_k']).assign(n_folds=FOLDS).to_csv(directory/'mean_fold_metrics.csv',index=False)
    grouped[METRICS].std().reset_index().to_csv(directory/'sd_fold_metrics.csv',index=False)
    prediction_metrics(predictions).merge(effective,on=['method','requested_k']).to_csv(directory/'pooled_oof_metrics.csv',index=False)


def main():
    baseline_signature=open_protocol(OUTPUT,'cv_baselines')
    native_signature=open_protocol(PROVENANCE/'native_run','native')
    for dataset in DATASETS:
        samples=pd.read_csv(SOURCE/dataset/'samples.csv',dtype={'sample_id':str})
        folds=make_folds(samples.y.to_numpy(int))
        # Completed folds validate without loading abundance matrices or fitting anything.
        pending=any(not complete(OUTPUT/dataset/f'fold_{f}/complete.json',baseline_signature) or
                    not complete(NATIVE_CV/dataset/f'fold_{f}/complete.json',native_signature) for f in range(FOLDS))
        if pending:
            raw=sparse.load_npz(SOURCE/dataset/'X_valid.npz').tocsr()
            with np.load(SOURCE/dataset/'edges.npz') as saved: pb,pg=saved['pair_bacteria'],saved['pair_gene']
            validate_samples(raw,samples,pb,pg)
            for fold,(a,b) in enumerate(folds):
                fold_run(dataset,fold,raw,samples,pb,pg,a,b,SOURCE/dataset/'X_valid.npz',baseline_signature)
                prediction_phase(dataset,fold,raw,samples,pb,pg,a,b,native_signature)
            del raw
            summarize_dataset(dataset,samples,OUTPUT,list(METHODS))
            summarize_dataset(dataset,samples,NATIVE_CV,NATIVE)
        print(f'Complete CV: {dataset}',flush=True)
