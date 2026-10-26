"""Bins e-value CV from frozen fold-local raw pairs; never construct graphs."""
from pathlib import Path
import hashlib
import json
import os
import sys
import time

for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import sparse
from experiments.cache import (PROJECT,CACHE,SOURCE,CV,NATIVE_CV,PROVENANCE,DATASETS,K_VALUES,
    file_hash,write_json,complete,seal,open_protocol)
from experiments.cross_validation import make_folds,evaluate_selection,summarize_dataset
from scripts import search_evalue_power as bins

OUTPUT=CACHE/'cv/bins'
METHOD='exact_categorical'


def training_selection(features,ids,training,y,family_size):
    """Validation values and labels never enter bin thresholds or evidence."""
    training=np.asarray(training);ids=np.asarray(ids);y=np.asarray(y)
    if (features.shape!=(len(y),len(ids)) or training.ndim!=1 or not len(training) or
            len(np.unique(training))!=len(training) or np.any(training<0) or np.any(training>=len(y))):
        raise ValueError('Misaligned training-only selection inputs')
    train=features[training].tocsc()
    if train.shape[1] and np.any(train.max(axis=0).toarray()<=train.min(axis=0).toarray()):
        raise ValueError('Cached training family contains constants')
    scores=bins.evidence(bins.prepare_bins(train),y[training][None])[0]
    native=bins.select(scores,ids,family_size)
    order=np.lexsort((ids,-scores))
    return dict(order=order,native=native,log_e=scores)


def fold_run(dataset,fold,samples,a,b,family_size,source_signature):
    source=NATIVE_CV/dataset/f'fold_{fold}'
    raw=source/'raw_pairs'; directory=OUTPUT/dataset/f'fold_{fold}'
    if not complete(source/'complete.json',source_signature):raise ValueError('Missing original CV fold')
    files=[raw/n for n in ('features.npz','preprocessing.npz','preprocessing_complete.json')]
    files += [source/'outer_split.npz',CV/dataset/f'fold_{fold}/split.npz',
              SOURCE/dataset/'samples.csv',SOURCE/dataset/'edges.npz']
    code=[Path(__file__),PROJECT/'scripts/search_evalue_power.py',PROJECT/'metrics/prediction.py',
          PROJECT/'experiments/cross_validation.py',PROJECT/'experiments/cache.py']
    protocol=dict(version='bins_three_bin_lambda1_cv_v1',dataset=dataset,fold=fold,k=K_VALUES,
        family_size=family_size,source_signature=source_signature,
        preprocessing='cached training-local prevalence and constants; supplied values unchanged',
        selection='training-only three bins; fixed lambda=1; exact normalized log-evidence; ties by pair ID',
        prediction='unscreened Top-K; training StandardScaler(with_mean=False); balanced L2 C=1; seed=42',
        versions=dict(python=sys.version,numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__),
        code={str(p.relative_to(PROJECT)):file_hash(p) for p in code},
        inputs={str(p.relative_to(PROJECT)):file_hash(p) for p in files})
    signature=hashlib.sha256(json.dumps(protocol,sort_keys=True).encode()).hexdigest()
    directory.mkdir(parents=True,exist_ok=True);path=directory/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise ValueError('Changed Bins CV checkpoint')
    if not path.exists():write_json(path,protocol)
    if complete(directory/'complete.json',signature):return
    for split in files[3:5]:
        with np.load(split) as saved:
            np.testing.assert_array_equal(saved['train_index'],a);np.testing.assert_array_equal(saved['test_index'],b)
    with np.load(raw/'preprocessing.npz') as saved:
        ids=saved['feature_ids']
        np.testing.assert_array_equal(saved['fit_indices'],a)
        if len(saved['evidence_indices']) or int(saved['prevalence_cutoff'])!=max(5,int(np.ceil(.05*len(a)))):
            raise ValueError('Wrong training-local preprocessing')
    features=sparse.load_npz(raw/'features.npz').tocsr()
    start=time.monotonic();marker=directory/'selection_complete.json'
    if complete(marker,signature):
        with np.load(directory/'selection.npz') as saved:selected={k:saved[k] for k in saved.files}
    else:
        selected=training_selection(features,ids,a,samples.y.to_numpy(int),family_size)
        np.savez_compressed(directory/'selection.npz',**selected)
        seal(marker,signature,[directory/'selection.npz'])
    print(f'{dataset}/fold_{fold}: {len(a)} training, {len(b)} validation; {len(ids):,} candidates; '
          f'{len(selected["native"]):,} discoveries; selection {time.monotonic()-start:.3f}s',flush=True)
    evaluate_selection(directory,dict(features=features,feature_ids=ids),selected,METHOD,dataset,fold,samples,a,b,signature)
    seal(directory/'complete.json',signature,[p for p in directory.iterdir() if p.is_file() and p.name!='complete.json'])
    bins.normalizer.cache_clear()


def main():
    source_signature=open_protocol(PROVENANCE/'native_run','native')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    for dataset in DATASETS:
        samples=pd.read_csv(SOURCE/dataset/'samples.csv',dtype={'sample_id':str})
        if samples.sample_id.duplicated().any() or ('subject_id' in samples and samples.subject_id.duplicated().any()):
            raise ValueError('Repeated participants')
        with np.load(SOURCE/dataset/'edges.npz') as saved:family_size=len(saved['pair_gene'])
        for fold,(a,b) in enumerate(make_folds(samples.y.to_numpy(int))):
            write_json(OUTPUT/'status.json',dict(status='running',dataset=dataset,fold=fold))
            fold_run(dataset,fold,samples,a,b,family_size,source_signature)
        summarize_dataset(dataset,samples,OUTPUT,[METHOD])
        print('Completed Bins CV:',dataset,flush=True)
    write_json(OUTPUT/'status.json',dict(status='complete',datasets=len(DATASETS),folds=5))


if __name__=='__main__':main()
