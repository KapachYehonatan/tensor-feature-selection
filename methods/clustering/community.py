"""Memory-bounded weighted modularity Leiden on exact Bernoulli edges."""
from __future__ import annotations
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
import json
import random
import shutil
import time
import numpy as np
from numba import njit
from methods.graphs.resources import GIB, Progress, ResourceMonitor, directory_bytes, peak_rss_bytes
from execution.cache import write_json
from methods.graphs.exact import iter_edge_shards

PART_DTYPE = np.dtype([("component", "<u4"), ("left", "<u4"), ("right", "<u4"), ("weight", "<f4")])
WORKSPACE_MULTIPLIER = 6

@njit(cache=True, nogil=True)
def _union_edges(parent, rank, left, right):
    """Union endpoint rows in-place; inputs are union-find and edge arrays."""
    for item in range(left.size):
        a, b = int(left[item]), int(right[item])
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        while parent[b] != b:
            parent[b] = parent[parent[b]]
            b = parent[b]
        if a == b:
            continue
        if rank[a] < rank[b]:
            a, b = b, a
        parent[b] = a
        if rank[a] == rank[b]:
            rank[a] += 1


@njit(cache=True, nogil=True)
def _compress_roots(parent):
    """Return a fully compressed root for every union-find row."""
    roots = np.empty(parent.size, dtype=np.uint32)
    for row in range(parent.size):
        root = row
        while parent[root] != root:
            root = parent[root]
        roots[row] = root
        parent[row] = root
    return roots


def _weights(shard: np.ndarray) -> np.ndarray:
    """Use the absolute signed-phi weights of the exact Bernoulli graph."""
    return np.abs(np.asarray(shard["phi"], dtype=np.float32))


def _leiden_component(task):
    """Fit one (component, left, right, weights, seed, resolution, iterations) task.

    Edge arrays have equal length; return original node rows, membership, and
    quality diagnostics. No participant labels enter this step.
    """
    component, left, right, weights, seed, resolution, iterations = task
    import igraph as ig

    keys = (left.astype(np.uint64) << np.uint64(32)) | right.astype(np.uint64)
    order = np.argsort(keys, kind="stable")
    keys, weights = keys[order], weights[order]
    starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
    keys = keys[starts]
    weights = np.maximum.reduceat(weights, starts)
    left = (keys >> np.uint64(32)).astype(np.uint32)
    right = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    vertices = np.unique(np.concatenate((left, right)))
    edges = np.column_stack((np.searchsorted(vertices, left), np.searchsorted(vertices, right)))
    graph = ig.Graph(n=int(vertices.size), edges=edges, directed=False)
    graph.es["weight"] = weights.tolist()
    ig.set_random_number_generator(random.Random(seed + int(component)))
    partition = graph.community_leiden(
        objective_function="modularity", weights="weight", resolution=resolution,
        n_iterations=iterations,
    )
    return (
        vertices, np.asarray(partition.membership, dtype=np.int64),
        float(partition.modularity), float(weights.sum()), int(keys.size),
    )


def _node_rows(ids: np.ndarray, order: np.ndarray, sorted_ids: np.ndarray) -> np.ndarray:
    """Map feature identifiers from shards to node-table row numbers."""
    places = np.searchsorted(sorted_ids, ids)
    if np.any(places >= sorted_ids.size) or np.any(sorted_ids[places] != ids):
        raise ValueError("edge references a feature absent from nodes.npy")
    return order[places].astype(np.uint32)


def run_community_detection(
    graph_dir: Path,
    seed: int = 17_729,
    resolution: float = 1.0,
    memory_limit_gib: float = 125.0,
    output_limit_gib: float = 100.0,
    free_disk_reserve_gib: float = 10.0,
    quiet: bool = False,
    workers: int = 1,
    working_memory_gib: float = 125.0,
    output_dir: Path | None = None,
    leiden_iterations: int = -1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit saved exact edges, return pair-aligned labels, and write the backend summary.

    Return (original pair IDs, aligned membership, summary); no text roundtrip.
    Weights are absolute phi. No feature values
    or labels enter partition fitting. Component IDs are stable under shard order.
    """
    if leiden_iterations == 0 or leiden_iterations < -1:
        raise ValueError("Invalid Leiden iteration count")
    output_dir = graph_dir if output_dir is None else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if min(memory_limit_gib, output_limit_gib, working_memory_gib) <= 0 or workers < 1:
        raise ValueError("community resource limits and workers must be positive")
    run = json.loads((graph_dir / "run.json").read_text())
    if run.get("status") != "complete":
        raise ValueError("Leiden requires a complete graph manifest")
    monitor = ResourceMonitor(output_dir, int(memory_limit_gib * GIB), int(output_limit_gib * GIB), int(free_disk_reserve_gib * GIB))
    progress = Progress(quiet)
    started = time.perf_counter()
    nodes = np.load(graph_dir / "nodes.npy", mmap_mode="r", allow_pickle=False)
    eligible = np.asarray(nodes["eligible"], dtype=bool)
    if run.get("edge_weight") != dict(field="phi", transform="abs"):
        raise ValueError("Leiden requires signed-phi edges with absolute weights")
    id_order = np.argsort(nodes["feature_id"])
    sorted_ids = np.asarray(nodes["feature_id"])[id_order]
    parent = np.arange(nodes.size, dtype=np.uint32)
    rank = np.zeros(nodes.size, dtype=np.uint8)
    edge_rows = np.zeros(nodes.size, dtype=bool)
    total_weight = 0.0
    edge_count = 0

    # First pass finds connected components with O(p) resident state.
    progress.show("leiden-components", force=True, edges=run.get("edge_count", 0))
    for shard in iter_edge_shards(graph_dir):
        left = _node_rows(np.asarray(shard["feature_j"]), id_order, sorted_ids)
        right = _node_rows(np.asarray(shard["feature_k"]), id_order, sorted_ids)
        _union_edges(parent, rank, left, right)
        edge_rows[left] = True
        edge_rows[right] = True
        total_weight += float(_weights(shard).sum(dtype=np.float64))
        edge_count += int(shard.size)
        progress.show("leiden-components", completed=edge_count, total=run.get("edge_count", edge_count))
    roots = _compress_roots(parent)
    component_roots, inverse = np.unique(roots[edge_rows], return_inverse=True)
    minima = np.full(component_roots.size, np.iinfo(np.uint32).max, dtype=np.uint32)
    np.minimum.at(minima, inverse, nodes["feature_id"][edge_rows])
    # Concurrent shard emission must not change component seeds or memberships.
    component_roots = component_roots[np.argsort(minima)]
    component_of_root = np.full(nodes.size, np.iinfo(np.uint32).max, dtype=np.uint32)
    component_of_root[component_roots] = np.arange(component_roots.size, dtype=np.uint32)

    # Second pass partitions edges into bounded temporary buckets.
    temporary = output_dir / ".community-tmp"
    shutil.rmtree(temporary, ignore_errors=True)
    temporary.mkdir()
    bucket_count = max(1, min(128, int(component_roots.size)))
    monitor.check_output(edge_count * PART_DTYPE.itemsize + 4096, "community partitions")
    written = 0
    try:
        for shard in iter_edge_shards(graph_dir):
            left = _node_rows(np.asarray(shard["feature_j"]), id_order, sorted_ids)
            right = _node_rows(np.asarray(shard["feature_k"]), id_order, sorted_ids)
            component = component_of_root[roots[left]]
            weights = _weights(shard)
            buckets = component % bucket_count
            for bucket in np.unique(buckets):
                take = buckets == bucket
                values = np.empty(int(take.sum()), dtype=PART_DTYPE)
                values["component"], values["left"], values["right"] = component[take], left[take], right[take]
                values["weight"] = weights[take]
                with (temporary / f"bucket-{int(bucket):03d}.bin").open("ab") as handle:
                    values.tofile(handle)
                written += int(values.size)
            progress.show("leiden-partition", completed=written, total=edge_count)

        membership = np.full(nodes.size, -1, dtype=np.int64)
        next_cluster = 0
        modularity = 0.0
        deduplicated_edges = 0
        paths = sorted(temporary.glob("bucket-*.bin"))
        pool = ProcessPoolExecutor(
            min(workers, int(component_roots.size)), mp_context=get_context("spawn")
        ) if workers > 1 and component_roots.size > 1 else None
        try:
            for bucket_number, path in enumerate(paths):
                estimated = path.stat().st_size * WORKSPACE_MULTIPLIER
                if estimated > working_memory_gib * GIB:
                    raise RuntimeError(
                        f"community bucket estimates {estimated / GIB:.2f} GiB, above "
                        f"the {working_memory_gib:.2f} GiB working-memory budget"
                    )
                monitor.check_memory(estimated, "community bucket")
                values = np.fromfile(path, dtype=PART_DTYPE)
                order = np.argsort(values["component"], kind="stable")
                values = values[order]
                starts = np.r_[0, np.flatnonzero(values["component"][1:] != values["component"][:-1]) + 1]
                stops = np.r_[starts[1:], values.size]
                tasks = []
                for start, stop in zip(starts, stops):
                    part = values[start:stop]
                    component_weight = float(part["weight"].sum(dtype=np.float64))
                    # Scale local modularity resolution to preserve the global graph objective.
                    tasks.append((
                        int(part["component"][0]), np.asarray(part["left"]), np.asarray(part["right"]),
                        np.asarray(part["weight"]), seed,
                        resolution * component_weight / total_weight if total_weight else resolution,
                        leiden_iterations,
                    ))
                results = list(pool.map(_leiden_component, tasks)) if pool is not None and len(tasks) > 1 else list(map(_leiden_component, tasks))
                for vertices, labels, local_q, component_weight, unique_edges in results:
                    membership[vertices] = labels + next_cluster
                    next_cluster += int(labels.max()) + 1
                    modularity += (component_weight / total_weight) * local_q if total_weight else 0.0
                    deduplicated_edges += unique_edges
                progress.show("leiden-optimize", completed=bucket_number + 1, total=len(paths), active_workers=min(workers, len(tasks)))
        finally:
            if pool is not None:
                pool.shutdown()

        isolates = np.flatnonzero((membership < 0) & eligible)
        membership[isolates] = np.arange(next_cluster, next_cluster + isolates.size)
        monitor.check_output(4096, "community summary")
        clustered = np.flatnonzero(membership >= 0)
        sizes = np.bincount(membership[clustered]) if clustered.size else np.empty(0, dtype=np.int64)
        summary = {
            "status": "complete", "method": run.get("method"), "edge_weight_field": "phi",
            "vertices": int(nodes.size), "eligible_vertices": int(eligible.sum()), "excluded_vertices": int((~eligible).sum()),
            "edge_vertices": int(edge_rows.sum()), "deduplicated_edges": deduplicated_edges,
            "connected_components": int(component_roots.size), "clusters": int(sizes.size),
            "singleton_clusters": int((sizes == 1).sum()),
            "singleton_fraction": float((sizes == 1).sum() / sizes.size) if sizes.size else 0.0,
            "eligible_features_in_singletons_fraction": float((sizes == 1).sum() / eligible.sum()) if eligible.any() else 0.0,
            "largest_cluster_size": int(sizes.max()) if sizes.size else 0,
            "modularity": modularity, "objective": "modularity", "resolution": float(resolution),
            "leiden_iterations": int(leiden_iterations),
            "seed": int(seed), "workers": int(workers),
            "wall_s": time.perf_counter() - started, "peak_rss_bytes": peak_rss_bytes(),
            "output_bytes": directory_bytes(output_dir),
        }
        write_json(output_dir / "community.json", summary)
        progress.show("leiden-finished", force=True, clusters=summary["clusters"], modularity=f"{modularity:.4f}")
        return np.asarray(nodes["feature_id"]).copy(), membership, summary
    finally:
        shutil.rmtree(temporary, ignore_errors=True)

