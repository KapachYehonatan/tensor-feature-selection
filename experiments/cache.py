"""Strict scientific checkpoints, independent of plotting, tests, and documentation."""
from pathlib import Path
import gzip
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import scipy
import sklearn

PROJECT = Path(__file__).resolve().parents[1]
CACHE = PROJECT/'outputs/.cache'
SOURCE = CACHE/'inputs'
CV = CACHE/'cv/baselines'
FULL = CACHE/'full_dataset/baselines'
NATIVE_CV = CACHE/'cv/native'
NATIVE_FULL = CACHE/'full_dataset/native'
PROVENANCE = CACHE/'provenance'
JOIN_PYTHON = Path('/home/kapachy/Microbium/bernoulli-correlation-join/.venv/bin/python')
DATASETS = ['HMP_2019_ibdmdb_unique', 'QinJ_2012', 'JieZ_2017', 'LiJ_2017',
            'NielsenHB_2014_unique', 'RubelMA_2020', 'WirbelJ_2018', 'ZhuF_2020']
K_VALUES = list(range(1, 11))+[20, 50, 75]
GRIDS = {'1-10': list(range(1, 11)), '10_20_50_75': [10, 20, 50, 75]}
METHODS = {'bacteria': 'Bacteria-collapsed', 'gene': 'Gene-collapsed', 'join': 'Bernoulli-join mean',
           'shoe': 'SHOE', 'mw_ebh': 'MW → e-BH', 'hurdle_product': 'Hurdle-product → e-BH',
           'hurdle_average': 'Hurdle-average → e-BH'}
NATIVE = list(METHODS)[3:]
METRICS = ['auroc', 'precision', 'recall', 'f1']
SEED = 42


def read_lines(path):
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'rt', encoding='utf-8') as handle:
        return [line.rstrip('\n') for line in handle]


def write_lines(path, values):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith('.gz') else open
    with opener(path, 'wt', encoding='utf-8') as handle:
        for value in values: handle.write(f'{value}\n')


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''): digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def complete(marker, signature):
    marker = Path(marker)
    if not marker.exists(): return False
    saved = json.loads(marker.read_text())
    if saved.get('signature') != signature: raise ValueError(f'Changed scientific protocol: {marker}')
    states = saved.get('files', saved.get('fold_cache_states', saved.get('cache_states')))
    if states is None: raise ValueError(f'Missing artifact manifest: {marker}')
    for name, size, mtime in states:
        path = marker.parent/name
        if not path.is_file() or (path.stat().st_size, path.stat().st_mtime_ns) != (size, mtime):
            raise ValueError(f'Changed scientific artifact: {path}')
    return True


def seal(marker, signature, files):
    marker = Path(marker)
    write_json(marker, dict(signature=signature, files=[
        (str(p.relative_to(marker.parent)), p.stat().st_size, p.stat().st_mtime_ns) for p in files]))


def build_protocol(stage):
    code = sorted(p for directory in ('methods', 'metrics', 'shuffled_fdp_test', 'experiments')
                  for p in (PROJECT/directory).rglob('*.py'))
    external = [JOIN_PYTHON.parents[2]/n for n in ('bernoulli_join.py', 'community_detection.py')]
    inputs = [SOURCE/d/n for d in DATASETS for n in ('X_valid.npz', 'samples.csv', 'edges.npz')]
    graph_versions = json.loads(subprocess.check_output([str(JOIN_PYTHON), '-c',
        'import json,igraph,numpy,sys; print(json.dumps(dict(igraph=igraph.__version__,numpy=numpy.__version__,python=sys.version)))'], text=True))
    protocol = dict(version='organized_supplied_values_v1', stage=stage, datasets=DATASETS,
        methods=METHODS, folds=5, seed=SEED, k_values=K_VALUES, alpha=.10,
        sample_normalization='none', abundance_transform='none', aggregation='mean supplied values',
        prevalence=[.05, 5], permutations=100, permutation_key='dataset/full_data',
        graph=dict(threshold=.70, adaptive_step=.05, edge_limit=500_000_000,
                   resolution=1., seed=17729, iterations=-1, objective='modularity',
                   workers=4, community_workers=1, memory_gib=200),
        classifier=dict(scaling='training StandardScaler(with_mean=False)', C=1.,
                        class_weight='balanced', solver='liblinear', max_iter=2000, seed=42),
        shoe=dict(split=.5, bins='zero and training-positive median', smoothing='Laplace',
                  evidence='single evidence block', weights='uniform'),
        mw_ebh_gamma=.5, hurdle='exact conditional normalizers; inverse null SD',
        correlation_limit=10000,
        versions=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,
                      pandas=pd.__version__,sklearn=sklearn.__version__,graph=graph_versions),
        code_sha256={str(p.relative_to(PROJECT)):file_hash(p) for p in code},
        external_code_sha256={str(p):file_hash(p) for p in external},
        input_sha256={str(p.relative_to(SOURCE)):file_hash(p) for p in inputs})
    signature = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return dict(signature=signature, **protocol)


def open_protocol(directory, stage):
    protocol = build_protocol(stage)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory/'protocol.json'
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError(f'Changed inputs/settings/scientific code: {path}; do not overwrite these results')
    else:
        if any(directory.iterdir()): raise ValueError(f'Nonempty cache has no validated protocol: {directory}')
        write_json(path, protocol)
    return protocol['signature']


def relocated_signature(directory, candidate):
    """Accept only an exact, validated input/code identity from this relocation."""
    record = PROVENANCE/'compatibility.json'
    if directory is not None and record.exists():
        saved = json.loads(record.read_text())
        path=Path(directory).resolve()
        key=str(path.relative_to(PROJECT)) if path.is_relative_to(PROJECT) else ''
        entry=saved['internal_signatures'].get(key)
        if entry and entry['new_signature']==candidate:
            for name, digest in saved['scientific_code_sha256'].items():
                if file_hash(PROJECT/name) != digest: raise ValueError('Changed code for relocated checkpoint')
            return entry['old_signature']
    return candidate
