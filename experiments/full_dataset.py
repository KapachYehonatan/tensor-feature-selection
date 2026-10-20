"""All-sample selection and discoveries, separate from prediction evaluation."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import pandas as pd
from scipy import sparse
from experiments.cache import (PROJECT, SOURCE, FULL as OUTPUT, NATIVE_FULL, PROVENANCE,
    DATASETS, NATIVE, K_VALUES, SEED, open_protocol, complete, seal, write_json, relocated_signature)
from methods.preprocessing import retained_pair_values, prevalence_mask, validate_samples
from methods.representations import collapsed_average, aggregate_pair_clusters
from methods.selectors.mann_whitney import mw_ranked_family
from methods.selectors import shoe, evalue as selectors
from methods.graphs.bernoulli_join import cluster
from metrics.correlation import average_pairwise_absolute_correlation
from shuffled_fdp_test.statistic import label_permutations, permutation_discoveries, discovery_statistics

METHODS = {'bacteria':'Bacteria-collapsed','gene':'Gene-collapsed','join':'Bernoulli-join mean'}
ALPHA, PERMUTATIONS, PERMUTATION_BATCH, SEED = .10, 100, 10, 42


def prepare_method(method, values, pb, pg, ids, labels, y, directory, signature):
    marker = directory/f'{method}_representation.json'
    if complete(marker, signature): return
    if method == 'join':
        features, _, sizes = aggregate_pair_clusters(values, sparse.csr_matrix((0, len(ids))), labels)
        feature_ids, level = np.arange(len(sizes)), 'pair_cluster'
    else:
        entity = pb if method == 'bacteria' else pg
        features, feature_ids = collapsed_average(values, entity[ids], int(entity.max())+1)
        sizes = np.bincount(entity[ids])[feature_ids]
        level = 'bacterium' if method == 'bacteria' else 'gene'
    total = features.shape[1]
    columns, table = mw_ranked_family(features, y, feature_ids, level, sizes, ALPHA)
    sparse.save_npz(directory/f'{method}_features.npz', features[:, columns].tocsr())
    table.to_csv(directory/f'{method}_tests.csv.gz', index=False)
    write_json(directory/f'{method}_counts.json', dict(samples=len(y), aggregated_features=total,
        tested_features=len(table), excluded_constants=total-len(table)))
    seal(marker, signature, [directory/f'{method}_{name}' for name in ('features.npz', 'tests.csv.gz', 'counts.json')])
    print(f'{directory.name}/{method}: {len(y)} samples, {total:,} predictors, {len(table):,} tested', flush=True)



def prepared(directory, raw, fit, evidence, signature):
    directory.mkdir(parents=True, exist_ok=True)
    marker = directory/'preprocessing_complete.json'
    if complete(marker, signature):
        with np.load(directory/'preprocessing.npz') as saved: p = {k:saved[k] for k in saved.files}
        expected_evidence = np.asarray(evidence) if evidence is not None else np.array([], int)
        if not np.array_equal(p['fit_indices'], fit) or not np.array_equal(p['evidence_indices'], expected_evidence):
            raise ValueError('Changed honest split')
        p.update({name:sparse.load_npz(directory/f'{name}.npz') for name in p['matrices']})
        return p
    start = time.monotonic()
    p = shoe.prepare(raw, fit, evidence) if evidence is not None else selectors.prepare(raw, fit)
    p['evidence_indices'] = np.asarray(evidence) if evidence is not None else np.array([], int)
    names = [name for name, value in p.items() if sparse.issparse(value)]
    p['matrices'] = np.asarray(names)
    for name in names: sparse.save_npz(directory/f'{name}.npz', p[name])
    np.savez_compressed(directory/'preprocessing.npz', **{k:v for k,v in p.items() if not sparse.issparse(v)})
    seal(marker, signature, [directory/f'{n}.npz' for n in names+['preprocessing']])
    print(f'{directory.parent.name}/{directory.name}: fit={len(fit)}, evidence={len(p["evidence_indices"])}, '
          f'retained={len(p["retained_pair_ids"]):,}, candidates={len(p["feature_ids"]):,}, '
          f'preprocessing={time.monotonic()-start:.3f}s', flush=True)
    return p


def select(directory, p, y, pb, pg, signature):
    marker = directory/'selection_complete.json'
    if complete(marker, signature):
        with np.load(directory/'selection.npz') as saved: return {k:saved[k] for k in saved.files}
    start = time.monotonic()
    log_e, contrasts = shoe.fit_evidence(p, y[None, :])
    order, native, log_a = shoe.native_selection(log_e[0], p['feature_ids'])
    result = dict(order=order, native=native, log_evidence=log_e[0], log_weighted_evidence=log_a)
    np.savez_compressed(directory/'selection.npz', **result)
    ids = p['feature_ids'][order]
    support = np.diff(p['evidence_bins'].indptr)[order]
    pd.DataFrame(dict(rank=np.arange(1, len(ids)+1), pair_id=ids, bacterium_id=pb[ids], gene_id=pg[ids],
        log_evidence=log_e[0, order], log_weighted_evidence=log_a[order],
        native_discovery=np.arange(len(ids)) < len(native), weight=1/len(ids) if len(ids) else 0.,
        positive_median=p['median'][order], one_positive_bin=p['one_bin'][order],
        training_zero_count=len(p['fit_indices'])-p['bin_counts'].reshape(-1, 2)[order].sum(axis=1),
        training_low_count=p['bin_counts'].reshape(-1, 2)[order, 0], training_high_count=p['bin_counts'].reshape(-1, 2)[order, 1],
        score_low=contrasts[0, order, 0], score_high=contrasts[0, order, 1], evidence_nonzeros=support)).to_csv(
            directory/'feature_evidence.csv.gz', index=False)
    write_json(directory/'selection_summary.json', dict(candidates=len(ids), discoveries=len(native),
        fit_samples=len(p['fit_indices']), evidence_samples=len(p['evidence_indices']),
        prevalence_cutoff=int(p['prevalence_cutoff']), retained_pairs=len(p['retained_pair_ids']),
        selection_seconds=time.monotonic()-start))
    seal(marker, signature, [directory/n for n in ['selection.npz', 'feature_evidence.csv.gz', 'selection_summary.json']])
    print(f'{directory.parent.name}/{directory.name}: native discoveries={len(native):,}; selection={time.monotonic()-start:.3f}s', flush=True)
    return result


def raw_selections(directory, p, targets, pb, pg, signature):
    x, labels = p['features'][p['fit_indices']], np.asarray(targets)[:, p['fit_indices']]
    identity = hashlib.sha256(signature.encode()+p['feature_ids'].tobytes()+p['fit_indices'].tobytes()+labels.tobytes()).hexdigest()
    identity = relocated_signature(directory, identity)
    marker = directory/'selection_complete.json'
    methods = NATIVE[1:]
    if complete(marker, identity):
        return {m:dict(np.load(directory/m/'selection.npz')) for m in methods}
    evidence, probabilities, effect = [], [], None
    for left in range(0, len(labels), PERMUTATION_BATCH):
        path = directory/f'mw_batch_{left}.npz'
        if path.exists():
            with np.load(path) as saved:
                if str(saved['signature']) != identity: raise ValueError('Changed MW checkpoint')
                log_e, pv, current_effect = saved['log_evidence'], saved['p_value'], saved['effect']
        else:
            log_e, pv, current_effect = selectors.mw_log_evidence(x, labels[left:left+PERMUTATION_BATCH])
            temporary = path.with_suffix('.tmp.npz')
            np.savez_compressed(temporary, signature=identity, log_evidence=log_e, p_value=pv, effect=current_effect)
            temporary.replace(path)
        evidence.append(log_e); probabilities.append(pv)
        if left == 0: effect = current_effect
        print(f'{directory.parent.name}: MW labels {min(left+PERMUTATION_BATCH,len(labels))}/{len(labels)}', flush=True)
    parts = selectors.hurdle_components(x, labels, directory/'hurdle_blocks')
    all_e = dict(mw_ebh=np.concatenate(evidence), **selectors.combine(parts))
    p_values = np.concatenate(probabilities)
    np.savez_compressed(directory/'components.npz', **{k:v[0] if v.ndim == 2 else v for k,v in parts.items()})
    result = {}
    for method in methods:
        case = directory/method; case.mkdir(exist_ok=True)
        log_e = all_e[method]
        order, native, weighted = shoe.native_selection(log_e[0], p['feature_ids'])
        result[method] = dict(order=order, native=native, log_evidence=log_e[0], log_weighted_evidence=weighted)
        np.savez_compressed(case/'selection.npz', **result[method])
        ids = p['feature_ids'][order]
        frame = pd.DataFrame(dict(rank=np.arange(1,len(ids)+1), pair_id=ids,
            bacterium_id=pb[ids], gene_id=pg[ids], log_evidence=log_e[0,order],
            log_weighted_evidence=weighted[order], native_discovery=np.arange(len(ids)) < len(native),
            weight=1/len(ids) if len(ids) else 0., support=parts['support'][order]))
        if method == 'mw_ebh':
            frame['p_value'], frame['mw_effect'] = p_values[0,order], effect[order]
        else:
            for field in ['log_presence', 'log_abundance', 'lambda_', 'eta']:
                frame[field.rstrip('_')] = parts[field][0,order]
            frame['present_variance'] = parts['present_variance'][order]
        frame.to_csv(case/'feature_evidence.csv.gz', index=False)
        counts = [len(shoe.native_selection(row, p['feature_ids'])[1]) for row in log_e[1:]]
        pd.DataFrame(dict(permutation=np.arange(len(counts)), discoveries=counts)).to_csv(case/'shuffled_counts.csv', index=False)
        write_json(case/'selection_summary.json', dict(candidates=len(ids), discoveries=len(native),
            fit_samples=len(p['fit_indices']), prevalence_cutoff=int(p['prevalence_cutoff']),
            retained_pairs=len(p['retained_pair_ids']), permutations=len(counts)))
        print(f'{directory.parent.name}/{method}: candidates={len(ids):,}; discoveries={len(native):,}', flush=True)
    seal(marker, identity, [f for f in directory.rglob('*') if f.is_file() and f != marker])
    return result


def pairwise_correlations(features,table):
    rows=[]
    for correction in ('bh','by'):
        columns=np.flatnonzero(table['significant_'+correction].to_numpy())
        rows.append(dict(correction=correction,selection='all_significant',requested_k=np.nan,
                         **average_pairwise_absolute_correlation(features,columns)))
    return pd.DataFrame(rows)


def analyze_method(method,targets,directory,signature):
    marker=directory/f'{method}_complete.json'
    if complete(marker,signature): return
    features=sparse.load_npz(directory/f'{method}_features.npz').tocsr()
    table=pd.read_csv(directory/f'{method}_tests.csv.gz',float_precision='round_trip')
    if features.shape != (targets.shape[1],len(table)): raise ValueError('Misaligned full-data representation')
    path=directory/f'{method}_shuffled_counts.csv'
    counts=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['permutation','correction','discoveries'])
    done=len(counts)//2
    expected={(i,c) for i in range(done) for c in ('bh','by')}
    if (counts.duplicated(['permutation','correction']).any() or done>len(targets) or
            set(map(tuple,counts[['permutation','correction']].to_numpy())) != expected or
            not counts.discoveries.between(0,len(table)).all() or not counts.discoveries.mod(1).eq(0).all()):
        raise ValueError('Invalid shuffled-label checkpoint')
    for start in range(done,len(targets),PERMUTATION_BATCH):
        batch=permutation_discoveries(features,targets[start:start+PERMUTATION_BATCH],ALPHA)
        batch['permutation']+=start
        counts=batch if counts.empty else pd.concat([counts,batch],ignore_index=True)
        temporary=path.with_suffix('.csv.tmp'); counts.to_csv(temporary,index=False); temporary.replace(path)
    pivot=counts.pivot(index='permutation',columns='correction',values='discoveries')
    if not pivot.by.le(pivot.bh).all(): raise ValueError('BY discoveries exceed BH')
    rows=[]
    for correction in ('bh','by'):
        values=counts[counts.correction.eq(correction)].sort_values('permutation').discoveries.to_numpy()
        rows.append(dict(method=method,correction=correction,samples=features.shape[0],
            **discovery_statistics(values,int(table['significant_'+correction].sum()),len(table))))
    pd.DataFrame(rows).to_csv(directory/f'{method}_statistics.csv',index=False)
    pairwise_correlations(features,table).assign(method=method).to_csv(directory/f'{method}_pairwise_correlations.csv',index=False)
    seal(marker,signature,[path,directory/f'{method}_statistics.csv',directory/f'{method}_pairwise_correlations.csv'])


def run_dataset(dataset,signature):
    directory,source=OUTPUT/dataset,SOURCE/dataset
    directory.mkdir(parents=True,exist_ok=True)
    if complete(directory/'complete.json',signature):
        if not complete(directory/'graph_complete.json',signature): raise ValueError('Missing graph checkpoint')
        return
    samples=pd.read_csv(source/'samples.csv',dtype={'sample_id':str})
    raw=sparse.load_npz(source/'X_valid.npz').tocsr()
    with np.load(source/'edges.npz') as saved: pb,pg=saved['pair_bacteria'],saved['pair_gene']
    validate_samples(raw,samples,pb,pg)
    y=samples.y.to_numpy(int)
    mask,_,cutoff=prevalence_mask(raw,.05,5)
    ids=np.flatnonzero(mask)
    if not len(ids): raise ValueError('All-sample prevalence retained no pair features')
    selection=directory/'sample_selection.npz'
    if selection.exists():
        with np.load(selection) as saved:
            for key,value in [('train_index',np.arange(len(y))),('feature_ids',ids),('prevalence_threshold',cutoff)]:
                if not np.array_equal(saved[key],value): raise ValueError('Changed all-sample selection')
    else:
        np.savez_compressed(selection,train_index=np.arange(len(y)),feature_ids=ids,prevalence_threshold=cutoff)
    if not complete(directory/'graph_complete.json',signature):
        labels,graph=cluster(source/'X_valid.npz',selection,directory/'join_graph')
        seal(directory/'graph_complete.json',signature,[p for p in (directory/'join_graph').rglob('*') if p.is_file()])
    else:
        labels,graph=cluster(source/'X_valid.npz',selection,directory/'join_graph')
    values=retained_pair_values(raw,ids)
    targets=label_permutations(y,dataset+'/full_data',PERMUTATIONS,SEED)
    target_path=directory/'shuffled_labels.npy'
    if target_path.exists(): np.testing.assert_array_equal(np.load(target_path),targets)
    else: np.save(target_path,targets,allow_pickle=False)
    for method in METHODS:
        prepare_method(method,values,pb,pg,ids,labels,y,directory,signature)
        analyze_method(method,targets,directory,signature)
    for name in ('statistics','pairwise_correlations'):
        pd.concat([pd.read_csv(directory/f'{m}_{name}.csv') for m in METHODS],ignore_index=True).assign(dataset=dataset).to_csv(directory/f'{name}.csv',index=False)
    write_json(directory/'graph_summary.json',dict(samples=len(y),retained_pairs=len(ids),clusters=graph['clusters'],
        requested_threshold=.70,effective_threshold=graph['effective_threshold'],raw_edges=graph['join']['edge_count'],
        final_edges=graph['community']['deduplicated_edges'],threshold_edge_counts=graph['threshold_edge_counts'],
        resource_retries=graph.get('resource_retries',[])))
    seal(directory/'complete.json',signature,[p for p in directory.iterdir() if p.is_file() and p.name not in ('complete.json','clustering.log')])


def selection_statistics(directory,p,selected,counts,method,dataset,samples):
    directory.mkdir(parents=True,exist_ok=True)
    if (len(counts)!=PERMUTATIONS or not np.array_equal(counts.permutation.to_numpy(),np.arange(PERMUTATIONS)) or
            not counts.discoveries.between(0,len(p['feature_ids'])).all() or not counts.discoveries.mod(1).eq(0).all()):
        raise ValueError('Incomplete or invalid permutation counts')
    native=selected['native']
    correlation=average_pairwise_absolute_correlation(p['features'],native)
    pd.DataFrame([dict(dataset=dataset,method=method,correction='native',selection='all_significant',
        requested_k=np.nan,**correlation)]).to_csv(directory/'pairwise_correlations.csv',index=False)
    stats=discovery_statistics(counts.discoveries.to_numpy(),len(native),len(p['feature_ids']))
    pd.DataFrame([dict(dataset=dataset,method=method,correction='native',samples=samples,**stats)]).to_csv(directory/'statistics.csv',index=False)
    counts.to_csv(directory/'shuffled_counts.csv',index=False)


def statistical_phase(dataset,raw,samples,pb,pg,signature):
    directory=NATIVE_FULL/dataset
    directory.mkdir(parents=True,exist_ok=True)
    if complete(directory/'complete.json',signature): return
    y=samples.y.to_numpy(int)
    targets=label_permutations(y,dataset+'/full_data',PERMUTATIONS,SEED)
    np.testing.assert_array_equal(np.load(OUTPUT/dataset/'shuffled_labels.npy'),targets)
    np.save(directory/'shuffled_labels.npy',targets)
    fit,evidence=shoe.honest_split(np.arange(len(y)),f'{dataset}/full_data')
    case=directory/'shoe'
    p=prepared(case,raw,fit,evidence,signature)
    selected=select(case,p,y,pb,pg,signature)
    path=case/'shuffled_counts.csv'
    counts=pd.read_csv(path) if path.exists() else pd.DataFrame(columns=['permutation','discoveries'])
    if (len(counts)>PERMUTATIONS or not np.array_equal(counts.permutation.to_numpy(),np.arange(len(counts))) or
            not counts.discoveries.between(0,len(p['feature_ids'])).all() or not counts.discoveries.mod(1).eq(0).all()):
        raise ValueError('Invalid SHOE permutation checkpoint')
    for start in range(len(counts),PERMUTATIONS,PERMUTATION_BATCH):
        log_e,_=shoe.fit_evidence(p,targets[start:start+PERMUTATION_BATCH])
        batch=pd.DataFrame([dict(permutation=start+i,discoveries=len(shoe.native_selection(row,p['feature_ids'])[1])) for i,row in enumerate(log_e)])
        counts=batch if counts.empty else pd.concat([counts,batch],ignore_index=True)
        temporary=path.with_suffix('.csv.tmp'); counts.to_csv(temporary,index=False); temporary.replace(path)
    selection_statistics(case,p,selected,counts,'shoe',dataset,len(y))
    del p
    p=prepared(directory/'raw_pairs',raw,np.arange(len(y)),None,signature)
    selections=raw_selections(directory/'raw_pairs',p,np.vstack((y,targets)),pb,pg,signature)
    for method,selected in selections.items():
        counts=pd.read_csv(directory/'raw_pairs'/method/'shuffled_counts.csv')
        selection_statistics(directory/method,p,selected,counts,method,dataset,len(y))
    for name in ('statistics','pairwise_correlations'):
        pd.concat([pd.read_csv(directory/m/f'{name}.csv') for m in NATIVE],ignore_index=True).to_csv(directory/f'{name}.csv',index=False)
    seal(directory/'complete.json',signature,[p for p in directory.rglob('*') if p.is_file() and p!=directory/'complete.json'])


def main():
    signature=open_protocol(OUTPUT,'full_baselines')
    native_signature=open_protocol(PROVENANCE/'native_run','native')
    for dataset in DATASETS:
        run_dataset(dataset,signature)
        if not complete(NATIVE_FULL/dataset/'complete.json',native_signature):
            samples=pd.read_csv(SOURCE/dataset/'samples.csv',dtype={'sample_id':str})
            raw=sparse.load_npz(SOURCE/dataset/'X_valid.npz').tocsr()
            with np.load(SOURCE/dataset/'edges.npz') as saved: pb,pg=saved['pair_bacteria'],saved['pair_gene']
            validate_samples(raw,samples,pb,pg)
            statistical_phase(dataset,raw,samples,pb,pg,native_signature)
            del raw
        print(f'Complete full dataset: {dataset}',flush=True)
