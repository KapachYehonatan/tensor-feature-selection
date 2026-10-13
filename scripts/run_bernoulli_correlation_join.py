#!/usr/bin/env python3
"""Exact Bernoulli-correlation join, cluster means, MW/BH and 80/20 prediction."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy import sparse
from research_utils import aggregate_pair_clusters, file_hash, JOIN_PYTHON, cache_signature
from scripts.run_method_comparison import (CACHE_VERSION, HOLDOUT_SEED, K_VALUES,
    MODEL_SEED, OUTPUT_ROOT, PREVALENCE_FRACTION, MINIMUM_SAMPLES, SCREEN_FDR,
    TEST_FRACTION, add_job_columns, bh_family, feature_table, evaluate_ranked,
    is_complete, real_jobs, save_method, split_and_normalize, load_job)

METHOD = 'bernoulli_correlation_join'
GRAPH_NAME = 'bernoulli_join_graph'
THRESHOLD = .70
JOIN_SOURCE = JOIN_PYTHON.parents[2]
WORKER = PROJECT/'scripts/bernoulli_join_worker.py'
WORKERS = 36
MEMORY_GIB, WORKING_MEMORY_GIB, COMMUNITY_WORKING_MEMORY_GIB = 125, 40, 125
OUTPUT_GIB, FREE_DISK_GIB, MEMORY_RESERVE_GIB, KEEP_GRAPH_GIB = 40, 500, 64, 20
SIDE_VERSION = 4


def signature(job, y, graph_only=False):
    source = Path(job['input'])
    files = [source/n for n in ('X_valid.npz', 'samples.csv', 'edges.npz')]
    code = [WORKER, PROJECT/'research_utils.py', Path(__file__),
            JOIN_SOURCE/'bernoulli_join.py', JOIN_SOURCE/'community_detection.py']
    settings = dict(version=SIDE_VERSION, method=METHOD, threshold=THRESHOLD,
        split=[TEST_FRACTION, HOLDOUT_SEED], prevalence=[PREVALENCE_FRACTION, MINIMUM_SAMPLES],
        inputs=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
        code={str(p):file_hash(p) for p in code})
    if not graph_only:
        settings.update(k=K_VALUES, screen_fdr=SCREEN_FDR, model_seed=MODEL_SEED,
                        model='Training-scaled balanced L2 C=1')
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(np.asarray(y, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def run_dataset(job):
    source = Path(job['input'])
    y = pd.read_csv(source/'samples.csv').y.to_numpy(int)
    run_signature, graph_signature = signature(job, y), signature(job, y, graph_only=True)
    destination = OUTPUT_ROOT/'real'/job['dataset']
    method_path, graph = destination/METHOD, destination/GRAPH_NAME
    marker = destination/f'{GRAPH_NAME}_complete.json'
    if is_complete(method_path, run_signature) and (graph/'assignments.npz').exists():
        print(f'Cached Bernoulli-join mean: {job["dataset"]}', flush=True)
        return
    if (method_path/'complete.json').exists():
        raise ValueError('Cached join inputs/code changed; use a new output directory')
    started = perf_counter()
    matrix, y, pb, pg, _ = load_job(job)
    a, b, mask, support, cutoff, train, test = split_and_normalize(matrix, y)
    ids = np.flatnonzero(mask).astype(np.int64)
    if not len(ids): raise ValueError('No training-retained pair features')
    destination.mkdir(parents=True, exist_ok=True)
    split_path = destination/f'{GRAPH_NAME}_split.npz'
    graph_ok = marker.exists() and (graph/'assignments.npz').exists()
    if graph_ok:
        graph_ok = json.loads(marker.read_text())['signature'] == cache_signature(marker, graph_signature)
        if not graph_ok: raise ValueError('Cached graph inputs changed; use a new output directory')
    if not graph_ok:
        if split_path.exists():
            with np.load(split_path) as saved:
                for key, expected in [('train_index', a), ('test_index', b), ('feature_ids', ids)]:
                    if not np.array_equal(saved[key], expected): raise ValueError(f'Cached {key} changed')
        else:
            np.savez_compressed(split_path, train_index=a, test_index=b, feature_ids=ids, prevalence_threshold=cutoff)
        command = [str(JOIN_PYTHON), str(WORKER), str(source/'X_valid.npz'), str(split_path), str(graph),
            '--threshold', str(THRESHOLD), '--workers', str(WORKERS), '--memory-gb', str(MEMORY_GIB),
            '--working-memory-gb', str(WORKING_MEMORY_GIB),
            '--community-working-memory-gb', str(COMMUNITY_WORKING_MEMORY_GIB),
            '--output-gb', str(OUTPUT_GIB), '--free-disk-gb', str(FREE_DISK_GIB),
            '--memory-reserve-gb', str(MEMORY_RESERVE_GIB), '--keep-graph-gb', str(KEEP_GRAPH_GIB)]
        subprocess.run(command, cwd=PROJECT, check=True)
        marker.write_text(json.dumps(dict(cache_version=CACHE_VERSION, signature=graph_signature), indent=2)+'\n')
    with np.load(graph/'assignments.npz') as saved:
        if not np.array_equal(saved['feature_ids'], ids): raise ValueError('Cached assignment IDs changed')
        labels = saved['clusters']
    info = json.loads((graph/'integration.json').read_text())
    if info.get('join_method', 'bernoulli') != 'bernoulli': raise ValueError('Not a Bernoulli graph')
    with np.load(split_path) as saved:
        for key, expected in [('train_index', a), ('test_index', b), ('feature_ids', ids)]:
            if not np.array_equal(saved[key], expected): raise ValueError(f'Graph {key} changed')
    t, v, sizes = aggregate_pair_clusters(train, test, labels)
    scores = bh_family(t, [y[a]])
    order = scores['order'][scores['q'][scores['order']] <= SCREEN_FDR]
    selected = feature_table(order, np.full(len(sizes), 'pair_cluster'), np.arange(len(sizes)), scores)
    selected['cluster_size'] = sizes[order]
    selected['similarity'] = 'bernoulli'
    selected['requested_similarity_threshold'] = THRESHOLD
    selected['effective_similarity_threshold'] = info['effective_threshold']
    ranking = selected.iloc[:max(K_VALUES)].copy()
    t, v = t[:, order[:max(K_VALUES)]], v[:, order[:max(K_VALUES)]]
    metrics, predictions = evaluate_ranked(METHOD, t, v, ranking, y[a], y[b],
        {'pair_cluster':len(sizes)}, {'pair_cluster':len(order)})
    metrics['requested_correlation_threshold'] = THRESHOLD
    metrics['effective_correlation_threshold'] = info['effective_threshold']
    add_job_columns(metrics, job)
    add_job_columns(predictions, job)
    save_method(method_path, run_signature, ranking, t, v, metrics, predictions, selected,
        perf_counter()-started, dict(requested_correlation_threshold=THRESHOLD,
                                    effective_correlation_threshold=info['effective_threshold']))


def main():
    for job in real_jobs(): run_dataset(job)
    from scripts.summarize_method_comparison import summarize
    from scripts.plot_fine_k_comparison import main as fine_plots
    summarize()
    fine_plots()


if __name__ == '__main__':
    main()
