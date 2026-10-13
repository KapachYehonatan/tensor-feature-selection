#!/usr/bin/env python3
"""Pack one sparse training split and run an external exact detection join."""

from pathlib import Path
import argparse
import csv
import gzip
import json
import os
import shutil
import sys

import numpy as np
from scipy import sparse


JOIN_PROJECT = Path("/home/kapachy/Microbium/bernoulli-correlation-join")
CHUNK_FEATURES = 10_000
GIB = 1024**3
COMMUNITY_SAFETY_FRACTION = 0.80
MAX_COMMUNITY_EDGES = 500_000_000
FALLBACK_STEP = 0.05


def available_memory_gib():
    fields = Path("/proc/meminfo").read_text(encoding="ascii").splitlines()
    value = next(line for line in fields if line.startswith("MemAvailable:"))
    return int(value.split()[1]) * 1024 / GIB


def pack_sparse_detection(matrix, feature_ids, output):
    """Write feature-major packed detection bits without densifying all columns."""
    matrix = sparse.csc_matrix(matrix, copy=True)
    matrix.eliminate_zeros()
    if matrix.data.size and np.min(matrix.data) < 0:
        raise ValueError("abundances must be nonnegative before detection binarization")
    feature_ids = np.asarray(feature_ids)
    n, p = matrix.shape
    if feature_ids.shape != (p,) or np.unique(feature_ids).size != p:
        raise ValueError("feature IDs must be unique and align with matrix columns")
    if np.any(feature_ids < 0) or np.any(feature_ids > np.iinfo(np.uint32).max):
        raise ValueError("feature IDs must fit in uint32")
    support = np.asarray(matrix.getnnz(axis=0)).ravel()
    if np.any(support == 0):
        raise ValueError("all-zero columns survived training prevalence filtering")
    output.mkdir(parents=True)
    word_count = (n + 63) // 64
    bits = np.lib.format.open_memmap(
        output / "bits.npy", mode="w+", dtype="<u8", shape=(p, word_count)
    )
    for start in range(0, p, CHUNK_FEATURES):
        stop = min(p, start + CHUNK_FEATURES)
        values = matrix[:, start:stop].toarray().T != 0
        packed = np.packbits(values, axis=1, bitorder="little")
        padded = np.zeros((stop - start, word_count * 8), dtype=np.uint8)
        padded[:, :packed.shape[1]] = packed
        bits[start:stop] = padded.view("<u8").reshape(stop - start, word_count)
    bits.flush()
    del bits
    np.save(output / "feature_ids.npy", feature_ids.astype("<u4"))
    metadata = {
        "format": "bernoulli-packed-uint64", "version": 1,
        "n": int(n), "p": int(p), "word_count": int(word_count),
        "bit_order": "little", "feature_ids": True,
    }
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return support


def stable_cluster_labels(feature_ids, labels, supports, n):
    """Give all-one columns singleton labels and make cluster IDs deterministic."""
    feature_ids = np.asarray(feature_ids)
    labels = np.asarray(labels, dtype=np.int64).copy()
    supports = np.asarray(supports)
    if np.any(supports == 0):
        raise ValueError("all-zero columns cannot be clustered")
    missing = labels < 0
    if np.any(missing & (supports != n)):
        raise ValueError("a nonconstant prevalence-retained feature lacks a cluster")
    next_label = int(labels[~missing].max()) + 1 if np.any(~missing) else 0
    for row in np.flatnonzero(missing):
        labels[row] = next_label
        next_label += 1
    clusters = sorted(np.unique(labels), key=lambda value: int(feature_ids[labels == value].min()))
    mapping = {old: new for new, old in enumerate(clusters)}
    return np.asarray([mapping[value] for value in labels], dtype=np.int64)


def fallback_thresholds(start):
    """Return deterministic thresholds from the requested value through one."""
    values = [round(float(start), 2)]
    candidate = np.ceil((float(start) + 1e-12) / FALLBACK_STEP) * FALLBACK_STEP
    while candidate <= 1.0 + 1e-12:
        rounded = round(float(candidate), 2)
        if rounded > values[-1]:
            values.append(rounded)
        candidate += FALLBACK_STEP
    return values


def choose_effective_threshold(counts, maximum_edges):
    """Choose the smallest threshold whose conservative workspace fits."""
    for threshold in sorted(counts):
        if counts[threshold] <= maximum_edges:
            return threshold
    raise RuntimeError(
        "even similarity threshold 1.0 exceeds the Leiden edge budget"
    )


def edge_strength(shard, manifest):
    """Return the exact similarity used as graph weight."""
    specification = manifest.get("edge_weight", {})
    field = specification.get(
        "field", manifest.get("edge_weight_field", "phi")
    )
    values = np.asarray(shard[field])
    if specification.get("transform", "identity") == "abs":
        values = np.abs(values)
    return values


def count_threshold_edges(graph, thresholds, manifest, iter_edge_shards):
    """Count all candidate thresholds in one streaming pass over exact edges."""
    frequencies = np.zeros(len(thresholds) + 1, dtype=np.int64)
    for shard in iter_edge_shards(graph):
        strength = edge_strength(shard, manifest)
        cutoffs = np.asarray(thresholds, dtype=strength.dtype)
        bins = np.searchsorted(cutoffs, strength, side="right")
        frequencies += np.bincount(bins, minlength=len(frequencies))
    return {
        float(threshold): int(frequencies[index + 1 :].sum())
        for index, threshold in enumerate(thresholds)
    }


def filter_exact_graph(source, destination, threshold, edge_count, manifest,
                       iter_edge_shards, directory_bytes, output_gib, free_disk_gib):
    """Stream an exact higher-threshold graph from an existing exact graph."""
    complete = destination / "run.json"
    if complete.exists():
        saved = json.loads(complete.read_text(encoding="utf-8"))
        if (
            saved.get("status") == "complete"
            and saved.get("threshold") == threshold
            and (not saved.get("edge_count") or (destination / "edges").exists())
        ):
            return saved
    shutil.rmtree(destination, ignore_errors=True)
    edges = destination / "edges"
    edges.mkdir(parents=True)
    os.link(source / "nodes.npy", destination / "nodes.npy")
    written_edges = 0
    shard_number = 0
    edge_itemsize = (
        int(manifest["edge_bytes"]) // int(manifest["edge_count"])
        if manifest.get("edge_count") else 12
    )
    fixed_bytes = (destination / "nodes.npy").stat().st_size
    for shard in iter_edge_shards(source):
        strength = edge_strength(shard, manifest)
        cutoff = np.asarray(threshold, dtype=strength.dtype)
        retained = np.asarray(shard)[strength >= cutoff]
        if retained.size:
            if shutil.disk_usage(destination).free - retained.nbytes < free_disk_gib * GIB:
                raise RuntimeError("filtered graph would violate the free-disk reserve")
            np.save(edges / f"part-{shard_number:06d}.npy", retained, allow_pickle=False)
            written_edges += int(retained.size)
            shard_number += 1
        if fixed_bytes + written_edges * edge_itemsize > output_gib * GIB:
            raise RuntimeError("filtered graph exceeded the output limit")
    if written_edges != edge_count:
        raise RuntimeError(
            f"filtered edge count changed during writing: {written_edges} != {edge_count}"
        )
    filtered = {
        **manifest,
        "status": "complete",
        "method": "exact-threshold-filter",
        "source_graph": str(source),
        "source_threshold": manifest["threshold"],
        "threshold": float(threshold),
        "edge_count": written_edges,
        "edge_shards": shard_number,
        "edge_bytes": written_edges * edge_itemsize,
        "output_bytes": directory_bytes(destination),
    }
    temporary = destination / "run.json.tmp"
    temporary.write_text(json.dumps(filtered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(complete)
    return filtered


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix", type=Path)
    parser.add_argument("split", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threshold", type=float, default=0.7)
    parser.add_argument(
        "--join-method", choices=("bernoulli", "jaccard"), default="bernoulli"
    )
    parser.add_argument("--workers", type=int, default=36)
    parser.add_argument("--community-workers", type=int, default=None,
                        help="Leiden workers; defaults to --workers for existing callers")
    parser.add_argument("--memory-gb", type=float, default=125.0)
    parser.add_argument("--working-memory-gb", type=float, default=40.0)
    parser.add_argument("--community-working-memory-gb", type=float, default=125.0)
    parser.add_argument("--output-gb", type=float, default=40.0)
    parser.add_argument("--free-disk-gb", type=float, default=500.0)
    parser.add_argument("--memory-reserve-gb", type=float, default=64.0)
    parser.add_argument("--keep-graph-gb", type=float, default=20.0)
    parser.add_argument("--community-max-edges", type=int, default=MAX_COMMUNITY_EDGES)
    parser.add_argument("--fixed-threshold", action="store_true",
                        help="Fail on resource limits instead of increasing the requested threshold")
    args = parser.parse_args()
    if args.community_max_edges < 1:
        raise ValueError("community edge cap must be positive")
    community_workers = args.workers if args.community_workers is None else args.community_workers
    if args.workers < 1 or community_workers < 1:
        raise ValueError("join and community workers must be positive")

    required = args.memory_gb + args.memory_reserve_gb
    available = available_memory_gib()
    if available < required:
        raise RuntimeError(
            f"need at least {required:.1f} GiB available before starting; "
            f"found {available:.1f} GiB"
        )
    free_disk = shutil.disk_usage(args.output.parent).free / GIB
    if free_disk < args.free_disk_gb:
        raise RuntimeError(
            f"need at least {args.free_disk_gb:.1f} GiB free before starting; "
            f"found {free_disk:.1f} GiB"
        )
    sys.path.insert(0, str(JOIN_PROJECT))
    from bernoulli_join import (
        directory_bytes, iter_edge_shards, load_packed_dataset, packed_supports,
        ResourceLimit, run_join,
    )
    from baseline_graphs import run_jaccard_join
    from community_detection import (
        PART_DTYPE, WORKSPACE_MULTIPLIER, run_community_detection,
    )

    with np.load(args.split) as split:
        train_index = split["train_index"]
        feature_ids = split["feature_ids"]
    if not len(feature_ids):
        raise RuntimeError("prevalence filtering retained no pair features")
    packed, graph = args.output / "packed", args.output / "graph"
    run_path = graph / "run.json"
    if args.output.exists():
        if not run_path.exists():
            raise RuntimeError("existing output has no completed exact-join manifest")
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        expected_method = "exact" if args.join_method == "bernoulli" else "jaccard"
        if (
            manifest.get("status") != "complete"
            or manifest.get("threshold", 1.0) > args.threshold
            or manifest.get("method") != expected_method
        ):
            raise RuntimeError("existing output is not a usable completed exact join")
        if manifest.get("edge_count", 0) and not (graph / "edges").exists():
            raise RuntimeError("existing exact graph has no retained edge shards")
        bits, packed_ids, metadata = load_packed_dataset(packed)
        if int(metadata["n"]) != len(train_index) or not np.array_equal(packed_ids, feature_ids):
            raise RuntimeError("existing packed input does not match the requested training split")
        supports = packed_supports(bits)
        print(f"Reusing completed exact graph with {manifest['edge_count']:,} edges", flush=True)
    else:
        args.output.mkdir(parents=True)
        matrix = sparse.load_npz(args.matrix).tocsr()[train_index][:, feature_ids]
        supports = pack_sparse_detection(matrix, feature_ids, packed)
        del matrix
        if args.join_method == "bernoulli":
            manifest = run_join(
                packed, graph, args.threshold, args.memory_gb, args.output_gb,
                args.free_disk_gb, args.workers, args.working_memory_gb,
                min_one_support=1,
            )
        else:
            manifest = run_jaccard_join(
                packed, graph, args.threshold, workers=args.workers,
                working_memory_gib=args.working_memory_gb,
                memory_limit_gib=args.memory_gb,
                output_limit_gib=args.output_gb,
                free_disk_reserve_gib=args.free_disk_gb,
                min_one_support=1,
            )
    if manifest["status"] != "complete":
        raise RuntimeError(f"exact join stopped: {manifest['status']}: {manifest.get('reason', '')}")

    thresholds = [args.threshold] if args.fixed_threshold else fallback_thresholds(args.threshold)
    memory_edge_budget = int(
        COMMUNITY_SAFETY_FRACTION * args.community_working_memory_gb * GIB
        / (PART_DTYPE.itemsize * WORKSPACE_MULTIPLIER)
    )
    maximum_edges = min(memory_edge_budget, args.community_max_edges)
    if manifest["threshold"] == args.threshold and manifest["edge_count"] <= maximum_edges:
        counts = {float(args.threshold): int(manifest["edge_count"])}
        effective_threshold = float(args.threshold)
        community_graph = graph
    else:
        counts = count_threshold_edges(graph, thresholds, manifest, iter_edge_shards)
        if args.fixed_threshold and counts[args.threshold] > maximum_edges:
            raise RuntimeError(f"Fixed threshold {args.threshold:.2f} requires {counts[args.threshold]:,} edges; "
                               f"configured/memory edge limit is {maximum_edges:,}")
        effective_threshold = choose_effective_threshold(counts, maximum_edges)
        if effective_threshold == manifest["threshold"]:
            community_graph = graph
        else:
            community_graph = args.output / f"graph-threshold-{effective_threshold:.2f}"
            filter_exact_graph(
                graph, community_graph, effective_threshold,
                counts[effective_threshold], manifest, iter_edge_shards,
                directory_bytes, args.output_gb, args.free_disk_gb,
            )
        print(
            f"Leiden fallback: requested={args.threshold:.2f}, "
            f"effective={effective_threshold:.2f}, edges={counts[effective_threshold]:,}, "
            f"edge_limit={maximum_edges:,} "
            f"(hard={args.community_max_edges:,}, memory={memory_edge_budget:,})",
            flush=True,
        )
    while True:
        try:
            community = run_community_detection(
                community_graph, seed=17_729, resolution=1.0,
                memory_limit_gib=args.memory_gb, output_limit_gib=args.output_gb,
                free_disk_reserve_gib=args.free_disk_gb, workers=community_workers,
                working_memory_gib=args.community_working_memory_gb,
            )
            break
        except (MemoryError, ResourceLimit, RuntimeError) as error:
            message = str(error).lower()
            if not isinstance(error, MemoryError) and not any(
                word in message for word in ("memory", "rss", "allocation")
            ):
                raise
            if args.fixed_threshold:
                raise RuntimeError(f"Fixed-threshold Leiden at {args.threshold:.2f} failed its memory guard") from error
            remaining = [value for value in thresholds if value > effective_threshold]
            if not remaining:
                raise
            if len(counts) < len(thresholds):
                counts = count_threshold_edges(
                    graph, thresholds, manifest, iter_edge_shards
                )
            effective_threshold = remaining[0]
            community_graph = args.output / f"graph-threshold-{effective_threshold:.2f}"
            filter_exact_graph(
                graph, community_graph, effective_threshold,
                counts[effective_threshold], manifest, iter_edge_shards,
                directory_bytes, args.output_gb, args.free_disk_gb,
            )
            print(
                f"Leiden memory guard ({type(error).__name__}) triggered; "
                "retrying from stored exact edges at "
                f"threshold={effective_threshold:.2f}, "
                f"edges={counts[effective_threshold]:,}",
                flush=True,
            )

    row_ids = np.empty(len(feature_ids), dtype=np.uint32)
    graph_labels = np.empty(len(feature_ids), dtype=np.int64)
    with gzip.open(community_graph / "clusters.csv.gz", "rt", encoding="utf-8") as handle:
        rows = csv.DictReader(handle)
        count = 0
        for count, row in enumerate(rows, start=1):
            if count > len(feature_ids):
                raise RuntimeError("join returned more assignments than retained features")
            row_ids[count - 1] = int(row["feature_id"])
            graph_labels[count - 1] = int(row["cluster"])
    if count != len(feature_ids):
        raise RuntimeError(
            f"join returned {count} assignments for {len(feature_ids)} retained features"
        )
    order = np.argsort(row_ids)
    requested_order = np.argsort(feature_ids)
    if not np.array_equal(row_ids[order], feature_ids[requested_order]):
        raise RuntimeError("join assignments do not match retained original feature IDs")
    restored = np.empty_like(graph_labels)
    restored[requested_order] = graph_labels[order]
    labels = stable_cluster_labels(feature_ids, restored, supports, len(train_index))
    sizes = np.bincount(labels)
    np.savez_compressed(
        args.output / "assignments.npz", feature_ids=feature_ids.astype(np.int64),
        clusters=labels, cluster_sizes=sizes,
    )
    with gzip.open(args.output / "cluster_assignments.csv.gz", "wt", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["feature_id", "cluster", "cluster_size", "training_support"])
        writer.writerows(
            (int(feature), int(cluster), int(sizes[cluster]), int(support))
            for feature, cluster, support in zip(feature_ids, labels, supports)
        )
    with (args.output / "cluster_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["cluster", "size"])
        writer.writerows((cluster, int(size)) for cluster, size in enumerate(sizes))

    graph_gib = directory_bytes(args.output) / GIB
    edges_pruned = graph_gib > args.keep_graph_gb
    if edges_pruned:
        for edge_directory in args.output.glob("graph*/edges"):
            shutil.rmtree(edge_directory)
    summary = {
        "status": "complete", "join_method": args.join_method,
        "requested_threshold": args.threshold,
        "effective_threshold": effective_threshold,
        "threshold_edge_counts": {str(key): value for key, value in counts.items()},
        "community_edge_budget": maximum_edges,
        "community_memory_edge_budget": memory_edge_budget,
        "community_hard_edge_cap": args.community_max_edges,
        "fixed_threshold": args.fixed_threshold,
        "community_safety_fraction": COMMUNITY_SAFETY_FRACTION,
        "features": int(len(feature_ids)), "clusters": int(len(sizes)),
        "singleton_clusters": int(np.sum(sizes == 1)),
        "all_one_singletons": int(np.sum(supports == len(train_index))),
        "graph_gib_before_pruning": graph_gib, "edges_pruned": edges_pruned,
        "join": manifest, "community": community,
    }
    (args.output / "integration.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
