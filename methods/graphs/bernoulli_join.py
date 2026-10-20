"""Fold/all-sample Bernoulli graph execution; fixed scientific resource guards."""
from pathlib import Path
import json
import subprocess
import numpy as np
from experiments.cache import PROJECT, JOIN_PYTHON

WORKER = PROJECT/'methods/graphs/worker.py'
THRESHOLD, EDGE_LIMIT = .70, 500_000_000
JOIN_WORKERS, COMMUNITY_WORKERS = 4, 1
WORKER_MEMORY_GIB, JOIN_WORKING_GIB, COMMUNITY_WORKING_GIB = 125, 24, 125
RESERVE_GIB, FREE_DISK_GIB, OUTPUT_GIB, KEEP_GRAPH_GIB = 16, 500, 40, 40


def cluster(matrix_path, selection_path, directory, threshold=THRESHOLD):
    marker = directory/'integration.json'
    if not marker.exists():
        manifest = directory/'graph/run.json'
        if directory.exists() and (not manifest.exists() or json.loads(manifest.read_text()).get('status') != 'complete'):
            number = 1
            archive = directory.with_name(directory.name+f'.failed-{number}')
            while archive.exists():
                number += 1
                archive = directory.with_name(directory.name+f'.failed-{number}')
            directory.rename(archive)
            print(f'Preserved incomplete graph in {archive}', flush=True)
        command = [str(JOIN_PYTHON), '-u', str(WORKER), str(matrix_path), str(selection_path), str(directory),
            '--threshold', str(threshold), '--workers', str(JOIN_WORKERS),
            '--community-workers', str(COMMUNITY_WORKERS), '--memory-gb', str(WORKER_MEMORY_GIB),
            '--working-memory-gb', str(JOIN_WORKING_GIB), '--community-working-memory-gb', str(COMMUNITY_WORKING_GIB),
            '--memory-reserve-gb', str(RESERVE_GIB), '--free-disk-gb', str(FREE_DISK_GIB),
            '--output-gb', str(OUTPUT_GIB), '--keep-graph-gb', str(KEEP_GRAPH_GIB),
            '--community-max-edges', str(EDGE_LIMIT)]
        with (directory.parent/'clustering.log').open('a') as log:
            subprocess.run(command, cwd=PROJECT, check=True, stdout=log, stderr=subprocess.STDOUT)
    info = json.loads(marker.read_text())
    community = info['community']
    if (info['status'] != 'complete' or info['fixed_threshold'] or info['edges_pruned']
            or info['requested_threshold'] != threshold or info['effective_threshold'] < threshold
            or community['deduplicated_edges'] > EDGE_LIMIT or info['community_hard_edge_cap'] != EDGE_LIMIT
            or round((info['effective_threshold']-threshold)/.05, 8) % 1 != 0
            or community['resolution'] != 1. or community['seed'] != 17729
            or community.get('objective', 'modularity') != 'modularity'
            or community.get('leiden_iterations', -1) != -1):
        raise ValueError('Clustering differs from the locked Bernoulli algorithm')
    with np.load(directory/'assignments.npz') as saved:
        labels, ids = saved['clusters'], saved['feature_ids']
    with np.load(selection_path) as saved:
        if not np.array_equal(ids, saved['feature_ids']): raise ValueError('Changed graph feature identities')
        if info['join']['n'] != len(saved['train_index']): raise ValueError('Changed graph training sample count')
    if (labels.shape != ids.shape or not np.issubdtype(labels.dtype, np.integer) or np.any(labels < 0)
            or np.any(np.bincount(labels) == 0) or len(np.bincount(labels)) != info['clusters']):
        raise ValueError('Invalid graph assignments')
    return labels, info
