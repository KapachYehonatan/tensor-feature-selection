#!/usr/bin/env python3
"""Private graph worker: unchanged edges, extended Leiden and small hierarchies."""
from pathlib import Path
import csv
import gzip
import json
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

JOIN_SOURCE = Path('/home/kapachy/Microbium/bernoulli-correlation-join')
RESOLUTIONS = (1., .75, .50, .25)
BATCH = 25
MIN_EXTRA = 100
MAX_EXTRA = 1000
PATIENCE = 3
QUALITY_TOL = 1e-10
LEIDEN_SEED = 17729
WORKERS = 4
MAX_ACTIVE_HIERARCHY = 5000
WORKING_GIB = 80


def canonical(labels):
    """Number communities by their first original feature, not library labels."""
    _, first, inverse = np.unique(labels, return_index=True, return_inverse=True)
    mapping = np.empty(len(first), dtype=np.int64)
    mapping[np.argsort(first)] = np.arange(len(first))
    return mapping[inverse]


def baseline_order(labels, baseline):
    """Preserve baseline ID tie order when an unchanged partition is returned."""
    labels = canonical(labels)
    count = int(labels.max())+1
    earliest = np.full(count, np.iinfo(np.int64).max, dtype=np.int64)
    first = earliest.copy()
    np.minimum.at(earliest, labels, baseline)
    np.minimum.at(first, labels, np.arange(len(labels)))
    mapping = np.empty(count, dtype=np.int64)
    mapping[np.lexsort((first, earliest))] = np.arange(count)
    return mapping[labels]


def changed_memberships(previous, current):
    """Count changed member sets, excluding arbitrary community renumbering."""
    pairs = np.unique(np.column_stack((previous, current)), axis=0)
    split = np.bincount(pairs[:, 0], minlength=int(previous.max())+1) > 1
    merged = np.bincount(pairs[:, 1], minlength=int(current.max())+1) > 1
    return int(np.sum(split[previous] | merged[current]))


def extended_partition(graph, resolution, component, minimum=MIN_EXTRA,
                       maximum=MAX_EXTRA, batch=BATCH, patience=PATIENCE,
                       tolerance=QUALITY_TOL):
    """One RNG stream across initial convergence and all warm-started batches."""
    import igraph as ig
    if minimum < 1 or maximum < minimum or batch < 1 or patience < 1:
        raise ValueError('Invalid iteration limits')
    ig.set_random_number_generator(random.Random(LEIDEN_SEED + int(component)))
    partition = graph.community_leiden(objective_function='modularity', weights='weight',
                                       resolution=resolution, n_iterations=-1)
    initial = canonical(partition.membership)
    previous, best = initial.copy(), initial.copy()
    quality = float(graph.modularity(previous, weights='weight', resolution=resolution))
    best_quality = quality
    trace = [dict(extra_iterations=0, quality=quality, gain=0., clusters=len(np.unique(previous)),
                  membership_changes=0, stable_batches=0)]
    stable, extra = 0, 0
    while extra < maximum:
        count = min(batch, maximum-extra)
        partition = graph.community_leiden(objective_function='modularity', weights='weight',
            resolution=resolution, n_iterations=count, initial_membership=previous.tolist())
        current = canonical(partition.membership)
        new_quality = float(graph.modularity(current, weights='weight', resolution=resolution))
        if not np.isfinite(new_quality) or new_quality < quality-tolerance:
            raise RuntimeError('Warm-started Leiden decreased modularity or produced nonfinite quality')
        gain = new_quality-quality
        extra += count
        stable = stable+1 if gain <= tolerance else 0
        trace.append(dict(extra_iterations=extra, quality=new_quality, gain=gain,
            clusters=len(np.unique(current)), membership_changes=changed_memberships(previous, current),
            stable_batches=stable))
        if graph.vcount() >= 1000:
            print(f'    component={component} vertices={graph.vcount():,} local_resolution={resolution:.6g} '
                  f'extra={extra} gain={gain:.3g} clusters={len(np.unique(current))}', flush=True)
        if new_quality > best_quality:
            best, best_quality = current.copy(), new_quality
        previous, quality = current, new_quality
        if extra >= minimum and stable >= patience:
            return initial, best, trace, True
    return initial, best, trace, False


def hierarchy_labels(baseline, left, right, weights, method, cutoff):
    """Max-edge community affinities; average/complete act on community units."""
    baseline = np.asarray(baseline, dtype=np.int64)
    if method not in ('average', 'complete') or not 0 < cutoff <= 1:
        raise ValueError('Invalid hierarchical linkage/cut')
    count = int(baseline.max())+1
    active = np.unique(np.r_[left, right]).astype(int)
    if len(active) > MAX_ACTIVE_HIERARCHY:
        raise RuntimeError('Hierarchy active-community bound exceeded; refusing dense allocation')
    mapping = np.arange(count)
    if not len(active):
        return baseline_order(baseline, baseline)
    similarity = np.zeros((len(active), len(active)))
    a, b = np.searchsorted(active, left), np.searchsorted(active, right)
    np.maximum.at(similarity, (a, b), weights)
    np.maximum.at(similarity, (b, a), weights)
    # Treat missing edges as unavailable; never join disconnected components.
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    _, components = connected_components(csr_matrix(similarity > 0), directed=False)
    for component in np.unique(components):
        take = np.flatnonzero(components == component)
        if len(take) < 2:
            continue
        distance = 1-similarity[np.ix_(take, take)]
        np.fill_diagonal(distance, 0.)
        tree = linkage(squareform(distance, checks=True), method=method)
        groups = fcluster(tree, t=1-cutoff, criterion='distance')
        for group in np.unique(groups):
            ids = active[take[groups == group]]
            mapping[ids] = ids.min()
    return baseline_order(mapping[baseline], baseline)


def _component(task):
    """Build igraph only once per original component, then vary resolution."""
    component, left, right, weights, total_weight = task
    import igraph as ig
    keys = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
    order = np.argsort(keys, kind='stable')
    keys, weights = keys[order], weights[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
    keys, weights = keys[starts], np.maximum.reduceat(weights, starts)
    left, right = (keys >> np.uint64(32)).astype(np.uint32), (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    vertices = np.unique(np.r_[left, right])
    edges = np.column_stack((np.searchsorted(vertices, left), np.searchsorted(vertices, right)))
    graph = ig.Graph(n=len(vertices), edges=edges, directed=False)
    graph.es['weight'] = weights.tolist()
    component_weight = float(weights.sum(dtype=np.float64))
    labels, traces, converged = {}, [], {}
    for resolution in RESOLUTIONS:
        # Match the existing component worker's global modularity normalization.
        local_resolution = resolution * component_weight/total_weight
        if len(vertices) >= 1000:
            print(f'  optimizing component={component} vertices={len(vertices):,} edges={len(weights):,} resolution={resolution:g}', flush=True)
        initial, final, trace, ok = extended_partition(graph, local_resolution, component)
        key = f'resolution_{resolution:g}'
        labels[key], converged[key] = final, ok
        if resolution == 1:
            labels['fresh_control'] = initial
            converged['fresh_control'] = True
        traces.extend(dict(component=int(component), resolution=resolution,
            component_vertices=len(vertices), component_weight=component_weight,
            local_resolution=local_resolution, converged=ok, **row) for row in trace)
    return vertices, labels, traces, converged


def write_json(path, value):
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run(request):
    sys.path.insert(0, str(JOIN_SOURCE))
    from community_detection import PART_DTYPE, _union_edges, _compress_roots, _node_rows, _weights
    from bernoulli_join import iter_edge_shards
    graph, output = Path(request['graph']), Path(request['output'])
    output.mkdir(parents=True, exist_ok=True)
    signature = request['signature']
    marker = output/'complete.json'
    if marker.exists():
        saved = json.loads(marker.read_text())
        if saved['signature'] != signature:
            raise ValueError('Incompatible graph-worker cache')
        return
    manifest = json.loads((graph/'run.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Graph is incomplete; no graph rebuild is permitted')
    with np.load(request['baseline']) as saved:
        ids, baseline = saved['feature_ids'], saved['clusters']
    if len(ids) != len(baseline) or len(np.unique(ids)) != len(ids) or np.any(baseline < 0):
        raise ValueError('Invalid baseline assignments')
    parent, rank = np.arange(len(ids), dtype=np.uint32), np.zeros(len(ids), dtype=np.uint8)
    id_order = np.argsort(ids)
    sorted_ids = ids[id_order]
    spec = manifest.get('edge_weight', {})
    field = spec.get('field', manifest.get('edge_weight_field', 'abs_phi'))
    transform = spec.get('transform', 'identity')
    edge_rows, total_weight, edge_count = np.zeros(len(ids), bool), 0., 0
    cross = {}
    started = time.monotonic()
    print(f'Preparing fixed graph {graph}; seed={LEIDEN_SEED}; workers={WORKERS}', flush=True)
    for number, shard in enumerate(iter_edge_shards(graph)):
        left = _node_rows(shard['feature_j'], id_order, sorted_ids)
        right = _node_rows(shard['feature_k'], id_order, sorted_ids)
        weight = _weights(shard, field, transform)
        if not np.isfinite(weight).all() or np.any(weight <= 0) or np.any(weight > 1+1e-6):
            raise ValueError('Expected positive baseline-transformed Bernoulli weights in (0,1]')
        _union_edges(parent, rank, left, right)
        edge_rows[left], edge_rows[right] = True, True
        total_weight += float(weight.sum(dtype=np.float64))
        edge_count += len(shard)
        a, b = baseline[left], baseline[right]
        different = a != b
        keys = (np.minimum(a[different], b[different]).astype(np.uint64) << np.uint64(32)) | np.maximum(a[different], b[different]).astype(np.uint64)
        if len(keys):
            order = np.argsort(keys, kind='stable')
            keys, values = keys[order], weight[different][order]
            starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1])+1]
            for key, value in zip(keys[starts], np.maximum.reduceat(values, starts)):
                cross[int(key)] = max(cross.get(int(key), 0.), float(value))
        if number % 20 == 0:
            print(f'  graph pass 1: {edge_count:,}/{manifest["edge_count"]:,} edges', flush=True)
    if edge_count != manifest['edge_count']:
        raise ValueError('Stored edge shards do not reproduce manifest edge count')
    roots = _compress_roots(parent)
    order = np.argsort(baseline, kind='stable')
    if np.any((baseline[order][1:] == baseline[order][:-1]) & (roots[order][1:] != roots[order][:-1])):
        raise ValueError('Cached baseline communities cross disconnected graph components')
    component_roots = np.unique(roots[edge_rows])
    component_of_root = np.full(len(ids), np.iinfo(np.uint32).max, dtype=np.uint32)
    component_of_root[component_roots] = np.arange(len(component_roots), dtype=np.uint32)
    if not len(component_roots):
        raise ValueError('No graph components')
    keys = np.array(sorted(cross), dtype=np.uint64)
    a, b = (keys >> np.uint64(32)).astype(int), (keys & np.uint64(0xFFFFFFFF)).astype(int)
    weight = np.array([cross[int(key)] for key in keys])
    settings = request['hierarchy']
    for setting in settings:
        labels = hierarchy_labels(baseline, a, b, weight, setting['linkage'], setting['cutoff'])
        np.savez_compressed(output/f'{setting["key"]}.npz', feature_ids=ids, clusters=labels)
    write_json(output/'graph_summary.json', dict(edge_count=edge_count, total_weight=total_weight,
        connected_components=len(component_roots), isolated_features=int((~edge_rows).sum()),
        minimum_connected_cluster_count=int((~edge_rows).sum())+len(component_roots),
        active_hierarchy_communities=len(np.unique(np.r_[a, b])), graph=str(graph)))

    # Temporary component buckets are shared across all resolutions, not copied per fit.
    temporary = output/'.component-buckets'
    prepared = temporary/'prepared.json'
    bucket_count = min(128, len(component_roots))
    if prepared.exists():
        if json.loads(prepared.read_text())['signature'] != signature:
            raise ValueError('Stale component preparation')
    else:
        temporary.mkdir(exist_ok=True)
        for path in temporary.glob('bucket-*.bin'):
            path.unlink()  # Only incomplete temporary buckets generated by this worker.
        estimated = edge_count*PART_DTYPE.itemsize
        if shutil.disk_usage(output).free < estimated+10*1024**3:
            raise RuntimeError('Insufficient disk for temporary graph component preparation')
        completed = 0
        for shard in iter_edge_shards(graph):
            left = _node_rows(shard['feature_j'], id_order, sorted_ids)
            right = _node_rows(shard['feature_k'], id_order, sorted_ids)
            component = component_of_root[roots[left]]
            weights = _weights(shard, field, transform)
            buckets = component % bucket_count
            for bucket in np.unique(buckets):
                take = buckets == bucket
                values = np.empty(int(take.sum()), dtype=PART_DTYPE)
                values['component'], values['left'], values['right'], values['weight'] = component[take], left[take], right[take], weights[take]
                with (temporary/f'bucket-{int(bucket):03d}.bin').open('ab') as handle:
                    values.tofile(handle)
            completed += len(shard)
            if completed % 20_000_000 < len(shard):
                print(f'  graph pass 2: {completed:,}/{edge_count:,} edges', flush=True)
        write_json(prepared, dict(signature=signature))
    checkpoints = output/'component_checkpoints'
    checkpoints.mkdir(exist_ok=True)
    names = ['fresh_control', *[f'resolution_{r:g}' for r in RESOLUTIONS]]
    membership = {name: np.full(len(ids), -1, dtype=np.int64) for name in names}
    next_cluster = {name: 0 for name in names}
    converged = {name: True for name in names}
    trace_paths = []
    with ProcessPoolExecutor(WORKERS, mp_context=get_context('spawn')) as pool:
        for bucket, path in enumerate(sorted(temporary.glob('bucket-*.bin'))):
            checkpoint = checkpoints/f'{path.stem}.npz'
            trace_path = checkpoints/f'{path.stem}.csv.gz'
            done = checkpoints/f'{path.stem}.json'
            if not done.exists():
                if path.stat().st_size*6 > WORKING_GIB*1024**3:
                    raise RuntimeError('Component bucket exceeds working-memory budget')
                values = np.fromfile(path, dtype=PART_DTYPE)
                values = values[np.argsort(values['component'], kind='stable')]
                starts = np.r_[0, np.flatnonzero(values['component'][1:] != values['component'][:-1])+1]
                stops = np.r_[starts[1:], len(values)]
                tasks = ((int(values['component'][start]), values['left'][start:stop],
                          values['right'][start:stop], values['weight'][start:stop], total_weight)
                         for start, stop in zip(starts, stops))
                rows, parts, local_next, traces = [], {name: [] for name in names}, {name: 0 for name in names}, []
                local_ok = {name: True for name in names}
                for vertices, labels, trace, ok in pool.map(_component, tasks, chunksize=1):
                    rows.append(vertices)
                    traces.extend(trace)
                    for name in names:
                        parts[name].append(labels[name]+local_next[name])
                        local_next[name] += int(labels[name].max())+1
                        local_ok[name] &= bool(ok[name])
                np.savez_compressed(checkpoint, rows=np.concatenate(rows), **{name: np.concatenate(parts[name]) for name in names})
                with gzip.open(trace_path, 'wt', newline='') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(traces[0]))
                    writer.writeheader()
                    writer.writerows(traces)
                write_json(done, dict(signature=signature, converged=local_ok))
                del values, rows, parts, traces
            info = json.loads(done.read_text())
            if info['signature'] != signature:
                raise ValueError('Stale component checkpoint')
            with np.load(checkpoint) as saved:
                rows = saved['rows']
                for name in names:
                    part = saved[name]
                    membership[name][rows] = part+next_cluster[name]
                    next_cluster[name] += int(part.max())+1
                    converged[name] &= bool(info['converged'][name])
            trace_paths.append(trace_path)
            print(f'  optimized bucket {bucket+1}/{bucket_count}; elapsed={time.monotonic()-started:.1f}s', flush=True)
    for name in names:
        missing = membership[name] < 0
        membership[name][missing] = np.arange(next_cluster[name], next_cluster[name]+int(missing.sum()))
        labels = baseline_order(membership[name], baseline)
        # Every community must lie wholly in one original connected component.
        sort = np.argsort(labels, kind='stable')
        if np.any((labels[sort][1:] == labels[sort][:-1]) & (roots[sort][1:] != roots[sort][:-1])):
            raise ValueError('Leiden merged disconnected graph components')
        np.savez_compressed(output/f'{name}.npz', feature_ids=ids, clusters=labels)
    with gzip.open(output/'convergence.csv.gz', 'wt', newline='') as handle:
        for index, path in enumerate(trace_paths):
            with gzip.open(path, 'rt') as source:
                header = next(source)
                if index == 0:
                    handle.write(header)
                shutil.copyfileobj(source, handle)
    write_json(marker, dict(status='complete', signature=signature, converged=converged,
        wall_seconds=time.monotonic()-started, resolutions=RESOLUTIONS,
        minimum_additional_iterations=MIN_EXTRA, maximum_additional_iterations=MAX_EXTRA))
    shutil.rmtree(temporary)  # Validated worker-owned temporary buckets only; source graph untouched.
    print(f'Graph clustering complete: {output}; convergence={converged}', flush=True)


if __name__ == '__main__':
    run(json.loads(Path(sys.argv[1]).read_text()))
