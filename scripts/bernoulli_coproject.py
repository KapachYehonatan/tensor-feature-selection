#!/usr/bin/env python3
"""Build and contract independent bacteria or gene Bernoulli graphs."""

from pathlib import Path
import hashlib
import json
import shutil

import numpy as np


GIB = 1024**3
VERSION = 2
BUCKETS = 256
SMALL_GRAPH_NODES = 4096
NODE_DTYPE = np.dtype([
    ("feature_id", "<u4"), ("column_index", "<u4"),
    ("support", "<u2"), ("minor_support", "<u2"),
    ("flipped", "?"), ("constant", "?"), ("eligible", "?"),
    ("exclusion_code", "u1"), ("pair_count", "<u8"),
])
EDGE_DTYPE = np.dtype([
    ("feature_j", "<u4"), ("feature_k", "<u4"),
    ("weight", "<f4"), ("numerator", "<f8"),
])
TEMP_DTYPE = np.dtype([("key", "<u8"), ("numerator", "<f8")])


def canonical_labels(entity_ids, labels):
    """Number communities by their smallest original biological identifier."""
    entity_ids = np.asarray(entity_ids, dtype=np.int64)
    labels = np.asarray(labels, dtype=np.int64)
    if entity_ids.shape != labels.shape or np.any(labels < 0):
        raise ValueError("entity IDs and nonnegative labels must align")
    groups = sorted(np.unique(labels), key=lambda value: int(entity_ids[labels == value].min()))
    mapping = {old: new for new, old in enumerate(groups)}
    return np.asarray([mapping[value] for value in labels], dtype=np.int64)


def contracted_ids(node_ids, labels):
    """Map every node to its community's smallest original identifier."""
    node_ids = np.asarray(node_ids, dtype=np.int64)
    labels = canonical_labels(node_ids, labels)
    minima = np.full(labels.max() + 1, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(minima, labels, node_ids)
    return minima[labels]


def _hash(*values):
    digest = hashlib.sha256()
    for value in values:
        if isinstance(value, np.ndarray):
            value = np.ascontiguousarray(value)
            digest.update(str(value.dtype).encode())
            digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
            digest.update(value.tobytes())
        else:
            digest.update(str(value).encode())
    return digest.hexdigest()


def _nodes(ids, counts):
    ids, counts = np.asarray(ids, dtype=np.int64), np.asarray(counts, dtype=np.uint64)
    if ids.shape != counts.shape or np.any(ids < 0) or np.any(ids > np.iinfo(np.uint32).max):
        raise ValueError("node IDs and pair counts must align and fit uint32")
    result = np.zeros(len(ids), dtype=NODE_DTYPE)
    result["feature_id"] = ids.astype(np.uint32)
    result["column_index"] = np.arange(len(ids), dtype=np.uint32)
    result["eligible"] = True
    result["pair_count"] = counts
    return result


def _denominators(left, right, count_lookup):
    left_count, right_count = count_lookup[left], count_lookup[right]
    result = left_count.astype(np.float64) * right_count
    loops = left == right
    result[loops] = left_count[loops].astype(np.float64) * (left_count[loops] - 1) / 2
    return result


def _write_edges(path, keys, numerators, count_lookup):
    left = (keys >> np.uint64(32)).astype(np.uint32)
    right = (keys & np.uint64(0xFFFFFFFF)).astype(np.uint32)
    denominator = _denominators(left, right, count_lookup)
    keep = denominator > 0
    edges = np.empty(int(np.sum(keep)), dtype=EDGE_DTYPE)
    edges["feature_j"], edges["feature_k"] = left[keep], right[keep]
    edges["numerator"] = numerators[keep]
    edges["weight"] = (numerators[keep] / denominator[keep]).astype(np.float32)
    np.save(path, edges, allow_pickle=False)
    return len(edges), edges.nbytes


def _dense_reduce(records, node_ids):
    size = len(node_ids)
    lookup = np.full(int(node_ids.max()) + 1, -1, dtype=np.int32)
    lookup[node_ids] = np.arange(size, dtype=np.int32)
    sums = np.zeros(size * size, dtype=np.float64)
    raw = 0
    for left, right, numerator in records:
        if not len(numerator):
            continue
        left_row, right_row = lookup[left], lookup[right]
        lo, hi = np.minimum(left_row, right_row), np.maximum(left_row, right_row)
        sums += np.bincount(lo.astype(np.int64) * size + hi, weights=numerator,
                            minlength=size * size)
        raw += len(numerator)
    positions = np.flatnonzero(sums)
    keys = ((node_ids[positions // size].astype(np.uint64) << np.uint64(32))
            | node_ids[positions % size].astype(np.uint64))
    return keys, sums[positions], raw


def _bucket_reduce(records, temporary, bucket_count, output_limit, free_disk):
    temporary.mkdir(parents=True)
    paths = [temporary / f"bucket-{index:04d}.bin" for index in range(bucket_count)]
    handles = [path.open("ab") for path in paths]
    raw = temporary_bytes = 0
    try:
        for record_number, (left, right, numerator) in enumerate(records, start=1):
            if not len(numerator):
                continue
            lo = np.minimum(left, right).astype(np.uint64)
            hi = np.maximum(left, right).astype(np.uint64)
            keys = (lo << np.uint64(32)) | hi
            buckets = ((keys ^ (keys >> np.uint64(32))) % np.uint64(bucket_count)).astype(np.int32)
            order = np.argsort(buckets, kind="stable")
            buckets, keys, numerator = buckets[order], keys[order], numerator[order]
            starts = np.r_[0, np.flatnonzero(buckets[1:] != buckets[:-1]) + 1]
            stops = np.r_[starts[1:], len(buckets)]
            for start, stop in zip(starts, stops):
                values = np.empty(stop - start, dtype=TEMP_DTYPE)
                values["key"], values["numerator"] = keys[start:stop], numerator[start:stop]
                values.tofile(handles[int(buckets[start])])
                temporary_bytes += values.nbytes
            raw += len(numerator)
            if temporary_bytes > output_limit:
                raise RuntimeError("projection temporary files exceeded the output limit")
            if record_number % 25 == 0:
                if shutil.disk_usage(temporary).free < free_disk:
                    raise RuntimeError("projection would violate the free-disk reserve")
                print(f"  streamed {record_number:,} shards; contributions={raw:,}", flush=True)
    finally:
        for handle in handles:
            handle.close()
    return paths, raw, temporary_bytes


def _write_bucket_edges(paths, edge_dir, count_lookup, output_limit):
    edge_count = edge_bytes = edge_shards = 0
    for path in paths:
        values = np.fromfile(path, dtype=TEMP_DTYPE)
        if not len(values):
            continue
        order = np.argsort(values["key"], kind="stable")
        keys = values["key"][order]
        numerators = values["numerator"][order]
        starts = np.r_[0, np.flatnonzero(keys[1:] != keys[:-1]) + 1]
        count, written = _write_edges(
            edge_dir / f"part-{edge_shards:06d}.npy", keys[starts],
            np.add.reduceat(numerators, starts), count_lookup)
        edge_count += count
        edge_bytes += written
        edge_shards += 1
        if edge_bytes > output_limit:
            raise RuntimeError("projected graph exceeded the output limit")
    return edge_count, edge_bytes, edge_shards


def _build_graph(destination, node_ids, node_counts, records, signature, metadata,
                 bucket_count=BUCKETS, output_limit_gib=40, free_disk_reserve_gib=500):
    destination = Path(destination)
    marker = destination / "run.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        if (saved.get("signature") == signature and saved.get("status") == "complete"
                and (saved.get("edge_count", 0) == 0 or (destination / "edges").exists())):
            return saved
    shutil.rmtree(destination, ignore_errors=True)
    edge_dir = destination / "edges"
    edge_dir.mkdir(parents=True)
    node_ids = np.asarray(node_ids, dtype=np.int64)
    node_counts = np.asarray(node_counts, dtype=np.uint64)
    np.save(destination / "nodes.npy", _nodes(node_ids, node_counts), allow_pickle=False)
    count_lookup = np.zeros(int(node_ids.max()) + 1, dtype=np.uint64)
    count_lookup[node_ids] = node_counts
    output_limit = int(output_limit_gib * GIB)
    free_disk = int(free_disk_reserve_gib * GIB)
    if shutil.disk_usage(destination).free < free_disk:
        raise RuntimeError("insufficient free disk for the configured reserve")
    if len(node_ids) <= SMALL_GRAPH_NODES:
        keys, sums, raw = _dense_reduce(records, node_ids)
        edge_count, edge_bytes = _write_edges(
            edge_dir / "part-000000.npy", keys, sums, count_lookup)
        edge_shards, temporary_bytes = 1, 0
    else:
        temporary = destination / ".projection-tmp"
        paths, raw, temporary_bytes = _bucket_reduce(
            records, temporary, bucket_count, output_limit, free_disk)
        try:
            edge_count, edge_bytes, edge_shards = _write_bucket_edges(
                paths, edge_dir, count_lookup, output_limit - temporary_bytes)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    manifest = {
        "status": "complete", "method": "bernoulli-contracted-projection",
        "architecture_version": VERSION, "signature": signature,
        "vertices": len(node_ids), "edge_count": edge_count,
        "edge_shards": edge_shards, "edge_bytes": edge_bytes,
        "contributing_edges": raw, "temporary_bytes": temporary_bytes,
        "edge_weight": {"field": "weight", "transform": "identity"},
        **metadata,
    }
    temporary_marker = marker.with_suffix(".json.tmp")
    temporary_marker.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary_marker.replace(marker)
    print(f"Built {metadata['dimension']} graph: {len(node_ids):,} nodes, "
          f"{edge_count:,} edges from {raw:,} contributions", flush=True)
    return manifest


def initial_graph(source, destination, pair_entity, feature_ids, sign_mode, dimension,
                  **limits):
    """Aggregate the fixed exact pair graph into one biological dimension."""
    source = Path(source)
    pair_entity = np.asarray(pair_entity, dtype=np.int64)
    feature_ids = np.asarray(feature_ids, dtype=np.int64)
    if sign_mode not in {"positive", "absolute"} or dimension not in {"bacteria", "gene"}:
        raise ValueError("unknown sign mode or dimension")
    run = json.loads((source / "run.json").read_text())
    if run.get("status") != "complete" or not (source / "edges").exists():
        raise ValueError("a completed stored pair graph is required")
    entities, counts = np.unique(pair_entity[feature_ids], return_counts=True)
    signature = _hash(VERSION, source.resolve(), (source / "run.json").stat().st_mtime_ns,
                      pair_entity, feature_ids, sign_mode, dimension)

    def records():
        for path in sorted((source / "edges").glob("part-*.npy")):
            shard = np.load(path, mmap_mode="r", allow_pickle=False)
            left_ids = np.asarray(shard["feature_j"], dtype=np.int64)
            right_ids = np.asarray(shard["feature_k"], dtype=np.int64)
            if len(left_ids) and max(left_ids.max(), right_ids.max()) >= len(pair_entity):
                raise ValueError("source graph references an unknown pair feature")
            phi = np.asarray(shard["phi"], dtype=np.float64)
            keep = phi > 0 if sign_mode == "positive" else np.ones(len(phi), dtype=bool)
            strength = phi[keep] if sign_mode == "positive" else np.abs(phi[keep])
            yield pair_entity[left_ids[keep]], pair_entity[right_ids[keep]], strength

    return _build_graph(
        destination, entities, counts, records(), signature,
        {"dimension": dimension, "sign_mode": sign_mode,
         "source_graph": str(source.resolve()), "source_edges": int(run["edge_count"]),
         "threshold": float(run["threshold"])}, **limits)


def contract_graph(source, destination, node_to_cluster, **limits):
    """Contract an aggregated graph without revisiting the fixed pair graph."""
    source = Path(source)
    nodes = np.load(source / "nodes.npy", mmap_mode="r", allow_pickle=False)
    node_ids = np.asarray(nodes["feature_id"], dtype=np.int64)
    node_to_cluster = np.asarray(node_to_cluster, dtype=np.int64)
    if node_to_cluster.shape != node_ids.shape:
        raise ValueError("one contracted ID is required for every current node")
    cluster_ids = np.unique(node_to_cluster)
    lookup = np.full(int(node_ids.max()) + 1, -1, dtype=np.int64)
    lookup[node_ids] = node_to_cluster
    cluster_counts = np.zeros(int(cluster_ids.max()) + 1, dtype=np.uint64)
    np.add.at(cluster_counts, node_to_cluster, np.asarray(nodes["pair_count"], dtype=np.uint64))
    counts = cluster_counts[cluster_ids]
    source_run = json.loads((source / "run.json").read_text())
    signature = _hash(VERSION, source_run["signature"], node_to_cluster)

    def records():
        for path in sorted((source / "edges").glob("part-*.npy")):
            shard = np.load(path, mmap_mode="r", allow_pickle=False)
            yield (lookup[np.asarray(shard["feature_j"], dtype=np.int64)],
                   lookup[np.asarray(shard["feature_k"], dtype=np.int64)],
                   np.asarray(shard["numerator"], dtype=np.float64))

    manifest = _build_graph(
        destination, cluster_ids, counts, records(), signature,
        {"dimension": source_run["dimension"], "sign_mode": source_run["sign_mode"],
         "source_graph": source_run["source_graph"],
         "parent_graph": str(source.resolve()), "threshold": source_run["threshold"]},
        **limits)
    if manifest["vertices"] > source_run["vertices"] or manifest["edge_count"] > source_run["edge_count"]:
        raise RuntimeError("graph contraction increased its node or edge count")
    return manifest
