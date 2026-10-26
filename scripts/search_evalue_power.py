"""Bins e-value: three-bin exp(lambda=1), exact null normalization, native e-BH.

The fixed raw-pair catalog is the family; feature-wise exclusions receive E=0.
"""
from pathlib import Path
import hashlib
import json
import os
import sys
import time
from functools import lru_cache

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import scipy
from scipy import sparse
from scipy.special import gammaln, logsumexp, xlogy
from experiments.cache import (PROJECT,CACHE,SOURCE,NATIVE_FULL,DATASETS,
                               file_hash,write_json,complete,seal,validate_fingerprints)

ROOT=CACHE/'evalue_power/bins'
ALPHA=.10


def log_choose(n,k):
    return gammaln(np.asarray(n)+1)-gammaln(np.asarray(k)+1)-gammaln(np.asarray(n)-k+1)


def log_lr(counts,positives,total_positive):
    counts=np.asarray(counts)
    positives=np.asarray(positives)
    total=counts.sum(axis=-1)
    fitted=(xlogy(positives,positives)+xlogy(counts-positives,counts-positives)-xlogy(counts,counts)).sum(axis=-1)
    null=xlogy(total_positive,total_positive)+xlogy(total-total_positive,total-total_positive)-xlogy(total,total)
    return np.maximum(fitted-null,0.)


def prepare_bins(matrix):
    x=sparse.csc_matrix(matrix,copy=True,dtype=float)
    x.sum_duplicates();x.eliminate_zeros();x.sort_indices()
    if not np.isfinite(x.data).all() or np.any(x.data<0): raise ValueError('Invalid supplied values')
    n,d=x.shape
    support=np.diff(x.indptr)
    bins=np.ones(x.nnz,np.int8)
    for j in range(d):
        a,b=x.indptr[j:j+2]
        if b>a: bins[a:b]+=x.data[a:b]>np.median(x.data[a:b])
    high_matrix=sparse.csc_matrix(((bins==2).astype(np.int8),x.indices.copy(),x.indptr.copy()),shape=x.shape)
    high_matrix.eliminate_zeros()
    presence=x.copy();presence.data=np.ones(x.nnz,np.int8)
    high=np.asarray(high_matrix.sum(axis=0)).ravel().astype(int)
    low=support-high
    return presence,high_matrix,np.column_stack((n-support,low,high))


def select(log_e,ids,family_size):
    log_e=np.asarray(log_e);ids=np.asarray(ids)
    if (log_e.ndim!=1 or log_e.shape!=ids.shape or family_size<len(ids) or
            len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=family_size) or
            np.isnan(log_e).any() or np.isposinf(log_e).any()):
        raise ValueError('Invalid fixed-family evidence')
    if not len(ids):return np.array([],int)
    order=np.lexsort((ids,-log_e))
    valid=np.flatnonzero(log_e[order]+np.log(np.arange(1,len(order)+1))>=np.log(family_size/ALPHA))
    return order[:int(valid[-1]+1)] if len(valid) else np.array([],int)



def table(n0,nl,nh,p):
    counts=np.array([n0,nl,nh],int);n=counts.sum()
    cells=np.array([(p-a-b,a,b) for a in range(min(nl,p)+1)
                    for b in range(min(nh,p)+1) if 0<=p-a-b<=n0])
    mass=log_choose(counts,cells).sum(axis=1)-log_choose(n,p)
    if abs(float(logsumexp(mass)))>1e-10: raise ValueError('Incomplete null support')
    hist=log_lr(counts,cells,p)
    return mass,hist


@lru_cache(None)
def normalizer(n0,nl,nh,p):
    mass,hist=table(n0,nl,nh,p)
    return float(logsumexp(mass+hist))


def evidence(prepared,targets):
    presence,high_matrix,counts=prepared
    targets=np.asarray(targets);n=presence.shape[0]
    if (targets.ndim!=2 or not len(targets) or targets.shape[1]!=n or
            not np.isin(targets,[0,1]).all() or np.any(targets.sum(axis=1)!=targets[0].sum())):
        raise ValueError('Expected aligned class-count-preserving labels')
    p=int(targets[0].sum())
    if p in (0,n): raise ValueError('Both classes required')
    if not len(counts): return np.empty((len(targets),0))
    present=np.asarray(targets.astype(np.int64)@presence)
    high=np.asarray(targets.astype(np.int64)@high_matrix)
    cells=np.stack((p-present,present-high,high),axis=-1)
    if np.any(cells<0) or np.any(cells>counts[None]): raise ValueError('Invalid observed cells')
    hist=log_lr(counts[None],cells,p)
    return hist-np.array([normalizer(*map(int,row),p) for row in counts])


def exact_checks():
    error=0.;maximum=0.
    for n in range(2,10):
        for n0 in range(n+1):
            for nl in range(n-n0+1):
                nh=n-n0-nl
                for p in range(1,n):
                    mass,hist=table(n0,nl,nh,p)
                    base=hist-normalizer(n0,nl,nh,p)
                    error=max(error,abs(float(np.exp(logsumexp(mass+base)))-1))
                    mean=float(np.exp(logsumexp(mass+base)))
                    maximum=max(maximum,mean)
    assert error<1e-10 and maximum<=1+1e-10
    result=dict(exact_null_mean_error=error,maximum_exact_null_mean=maximum,
                lambda_value=1.,bins=3)
    rng=np.random.default_rng(42);fdps=[];powers=[];nulls=[]
    for _ in range(300):
        y=np.tile([0,1],40)
        null=rng.binomial(1,.35,(80,9))*rng.lognormal(0,1,(80,9))
        signal=rng.binomial(1,np.where(y[:,None]==1,.8,.12),(80,10))*rng.lognormal(y[:,None]*1.5,1,(80,10))
        x=np.column_stack((np.repeat(null,10,axis=1),signal))
        ids=np.flatnonzero((np.sum(x>0,axis=0)>=5)&(np.ptp(x,axis=0)>0))
        log_e=evidence(prepare_bins(sparse.csr_matrix(x[:,ids])),y[None])[0]
        selected=ids[select(log_e,ids,100)]
        fdps.append(np.sum(selected<90)/max(1,len(selected)));powers.append(np.sum(selected>=90)/10)
        x=rng.binomial(1,.35,(80,10))*rng.lognormal(0,1,(80,10))
        x=np.repeat(x,10,axis=1)
        ids=np.flatnonzero((np.sum(x>0,axis=0)>=5)&(np.ptp(x,axis=0)>0))
        log_e=evidence(prepare_bins(sparse.csr_matrix(x[:,ids])),y[None])[0]
        nulls.append(bool(len(select(log_e,ids,100))))
    for name,values in (('mixed_null_fdr',fdps),('global_null_fdr',nulls)):
        result[name]=float(np.mean(values))
        result[name+'_mc_se']=float(np.std(values,ddof=1)/np.sqrt(len(values)))
        assert result[name]<ALPHA+4*result[name+'_mc_se']
    result.update(simulation_repetitions=len(fdps),strong_fixture_power=float(np.mean(powers)))
    print(result,flush=True);return result


def run():
    ROOT.mkdir(parents=True,exist_ok=True)
    protected=json.loads((CACHE/'provenance/compatibility.json').read_text())['scientific_code_sha256']
    validate_fingerprints(protected)
    protocol=dict(version='bins_three_bin_lambda1_selection_v2',alpha=ALPHA,lambda_value=1.,
        bins='absent, positive <= full-column positive median, positive > median',
        score='three-bin log-likelihood ratio; exp(T)/E0(exp(T)); exact conditional normalization',
        family='fixed complete supplied raw-pair catalog; excluded features E=0',
        assumptions='independent participants, marginal within-feature label exchangeability under each null',
        filtering='feature-wise prevalence and constants only; no across-feature label-dependent screening',
        per_dataset_tuning=False,
        code={str(Path(__file__).relative_to(PROJECT)):file_hash(__file__),
              'experiments/cache.py':file_hash(PROJECT/'experiments/cache.py')},
        protected_main_scientific_code=protected,
        software=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,pandas=pd.__version__))
    path=ROOT/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:
        raise ValueError('Incompatible search protocol; refusing to mix results')
    write_json(path,protocol)
    checks=ROOT/'implementation_checks.json'
    if not checks.exists(): write_json(checks,exact_checks())
    summary=[]
    for dataset in DATASETS:
        start=time.monotonic();dest=ROOT/dataset;dest.mkdir(exist_ok=True)
        case=NATIVE_FULL/dataset/'raw_pairs'
        sources=[case/n for n in ('features.npz','preprocessing.npz','preprocessing_complete.json')]
        sources += [SOURCE/dataset/n for n in ('samples.csv','edges.npz')]
        identity=dict(protocol=protocol,inputs={str(p.relative_to(PROJECT)):file_hash(p) for p in sources})
        key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        marker=dest/'complete.json'
        if marker.exists():
            complete(marker,key)
            summary.append(pd.read_csv(dest/'comparison.csv'));continue
        saved=json.loads((case/'preprocessing_complete.json').read_text())
        complete(case/'preprocessing_complete.json',saved['signature'])
        samples=pd.read_csv(SOURCE/dataset/'samples.csv')
        if samples.sample_id.duplicated().any() or ('subject_id' in samples and samples.subject_id.duplicated().any()):
            raise ValueError('Nonindependent participants')
        y=samples.y.to_numpy(int)
        with np.load(case/'preprocessing.npz') as saved:
            ids=saved['feature_ids'];fit=saved['fit_indices']
            if not np.array_equal(fit,np.arange(len(y))): raise ValueError('Not a full-data representation')
            if int(saved['prevalence_cutoff'])!=max(5,int(np.ceil(.05*len(y)))): raise ValueError('Wrong filtering')
        with np.load(SOURCE/dataset/'edges.npz') as saved: family_size=len(saved['pair_gene'])
        x=sparse.load_npz(case/'features.npz')
        if x.shape!=(len(y),len(ids)): raise ValueError('Misaligned feature cache')
        prepared=prepare_bins(x);del x
        print(dataset,'samples',len(y),'fixed family',family_size,'eligible',len(ids),flush=True)
        real=evidence(prepared,y[None])[0]
        discovery_ids=ids[select(real,ids,family_size)]
        np.savez_compressed(dest/'evidence.npz',feature_ids=ids,log_e=real,discovery_ids=discovery_ids)
        frame=pd.DataFrame([dict(dataset=dataset,samples=len(y),full_family=family_size,
                                 eligible_features=len(ids),discoveries=len(discovery_ids))])
        frame.to_csv(dest/'comparison.csv',index=False);summary.append(frame)
        seal(marker,key,[dest/'evidence.npz',dest/'comparison.csv'])
        saved=json.loads(marker.read_text())
        write_json(marker,dict(saved,input_identity=identity,seconds=time.monotonic()-start))
        pd.concat(summary,ignore_index=True).to_csv(ROOT/'comparison.csv',index=False)
        print('COMPLETE',dataset,frame.to_dict('records')[0],flush=True)
        del prepared
        normalizer.cache_clear()
    pd.concat(summary,ignore_index=True).to_csv(ROOT/'comparison.csv',index=False)
    print('ALL DATASETS COMPLETE',flush=True)


if __name__=='__main__': run()
