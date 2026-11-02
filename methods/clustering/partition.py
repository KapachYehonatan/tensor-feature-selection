"""Threshold budgets, partition fitting, and deterministic pair assignments."""
from decimal import Decimal, ROUND_FLOOR
from argparse import Namespace
from pathlib import Path
import json
import numpy as np
from methods.graphs.resources import GIB
from methods.graphs.preparation import count_threshold_edges, filter_exact_graph

COMMUNITY_SAFETY_FRACTION = .80

def stable_cluster_labels(feature_ids: np.ndarray, labels: np.ndarray,
                           supports: np.ndarray, n: int) -> np.ndarray:
    """Map pair-aligned labels to compact deterministic IDs; n is the training sample count.

    Constant all-one pairs have no graph assignment and receive separate labels.
    """
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
    clusters, inverse = np.unique(labels, return_inverse=True)
    minima = np.full(len(clusters), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(minima, inverse, feature_ids)
    mapping = np.empty(len(clusters), dtype=np.int64)
    mapping[np.argsort(minima)] = np.arange(len(clusters))
    return mapping[inverse]


def fallback_thresholds(start: float, step: float = .05) -> list[float]:
    """Return deterministic thresholds from the requested value through one."""
    if not np.isfinite(start) or not np.isfinite(step) or not 0 < start <= 1 or not 0 < step <= 1:
        raise ValueError("Invalid adaptive threshold grid")
    values = [float(start)]
    first, increment = Decimal(str(start)), Decimal(str(step))
    candidate = (first / increment).to_integral_value(rounding=ROUND_FLOOR) * increment + increment
    while candidate < 1:
        values.append(float(candidate))
        candidate += increment
    if values[-1] != 1.:
        values.append(1.)
    return values


def choose_effective_threshold(counts: dict[float, int], maximum_edges: int) -> float:
    """Choose the smallest threshold whose conservative workspace fits."""
    for threshold in sorted(counts):
        if counts[threshold] <= maximum_edges:
            return threshold
    raise RuntimeError(
        "even similarity threshold 1.0 exceeds the Leiden edge budget"
    )


def threshold_graph(args: Namespace, source: Path, manifest: dict, threshold: float, edge_count: int) -> Path:
    """Return the source or materialize its exact higher-threshold subset."""
    from methods.graphs.exact import iter_edge_shards
    from methods.graphs.resources import directory_bytes
    if threshold == manifest["threshold"]:
        return source
    destination = args.output / f"graph-threshold-{threshold!r}"
    filter_exact_graph(source, destination, threshold, edge_count, manifest,
                       iter_edge_shards, directory_bytes, args.output_gb, args.free_disk_gb)
    return destination


def fit_partition(args: Namespace, graph: Path, manifest: dict) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit Leiden, retrying only resource failures; return pair IDs, aligned membership and metadata.

    Every retry filters the same exact source edges. Validation scores and labels
    never influence threshold selection.
    """
    from methods.graphs.exact import iter_edge_shards
    from methods.graphs.resources import ResourceLimit
    from methods.clustering.community import PART_DTYPE, WORKSPACE_MULTIPLIER, run_community_detection
    community_workers = args.community_workers
    thresholds = [args.threshold] if args.fixed_threshold else fallback_thresholds(args.threshold, args.threshold_step)
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
        community_graph = threshold_graph(args, graph, manifest, effective_threshold, counts[effective_threshold])
        print(
            f"Leiden fallback: requested={args.threshold:.2f}, "
            f"effective={effective_threshold:.2f}, edges={counts[effective_threshold]:,}, "
            f"edge_limit={maximum_edges:,} "
            f"(hard={args.community_max_edges:,}, memory={memory_edge_budget:,})",
            flush=True,
        )
    resource_retries = []
    while True:
        try:
            row_ids, graph_labels, community = run_community_detection(
                community_graph, seed=args.seed, resolution=args.resolution,
                memory_limit_gib=args.memory_gb, output_limit_gib=args.output_gb,
                free_disk_reserve_gib=args.free_disk_gb, workers=community_workers,
                working_memory_gib=args.community_working_memory_gb,
                output_dir=args.output / 'community', leiden_iterations=args.iterations,
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
            resource_retries.append(dict(from_threshold=effective_threshold,
                to_threshold=remaining[0], reason=f'{type(error).__name__}: {error}'))
            effective_threshold = remaining[0]
            community_graph = threshold_graph(args, graph, manifest, effective_threshold, counts[effective_threshold])
            print(
                f"Leiden memory guard ({type(error).__name__}) triggered; "
                "retrying from stored exact edges at "
                f"threshold={effective_threshold:.2f}, "
                f"edges={counts[effective_threshold]:,}",
                flush=True,
            )

    return row_ids, graph_labels, dict(effective_threshold=effective_threshold, resource_retries=resource_retries,
                threshold_edge_counts={str(key): value for key, value in counts.items()},
                community_edge_budget=maximum_edges, community_memory_edge_budget=memory_edge_budget,
                community=community)


def write_assignments(output: Path, feature_ids: np.ndarray, supports: np.ndarray,
                      n_samples: int, row_ids: np.ndarray, graph_labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Validate/reorder backend IDs, restore all-one singletons, and save compact NPZ arrays."""
    if row_ids.shape != feature_ids.shape or graph_labels.shape != feature_ids.shape:
        raise RuntimeError("join assignment count does not match retained features")
    order = np.argsort(row_ids)
    requested_order = np.argsort(feature_ids)
    if not np.array_equal(row_ids[order], feature_ids[requested_order]):
        raise RuntimeError("join assignments do not match retained original feature IDs")
    restored = np.empty_like(graph_labels)
    restored[requested_order] = graph_labels[order]
    labels = stable_cluster_labels(feature_ids, restored, supports, n_samples)
    sizes = np.bincount(labels)
    np.savez_compressed(
        output / "assignments.npz", feature_ids=feature_ids.astype(np.int64),
        clusters=labels, cluster_sizes=sizes,
    )
    return labels, sizes


def cluster_graph(args: Namespace, train_index: np.ndarray, feature_ids: np.ndarray) -> None:
    """Validate the saved training graph, fit a partition, and export its records."""
    from methods.graphs.exact import load_packed_dataset, packed_supports
    from methods.graphs.resources import directory_bytes, ResourceMonitor
    args.output.mkdir(parents=True, exist_ok=True)
    graph = args.graph_source / "graph"
    manifest = json.loads((graph / "run.json").read_text())
    bits, packed_ids, metadata = load_packed_dataset(args.graph_source / "packed")
    if int(metadata["n"]) != len(train_index) or not np.array_equal(packed_ids, feature_ids):
        raise ValueError("Saved graph and requested fitting indices do not match")
    supports = packed_supports(bits)
    row_ids, graph_labels, fitting = fit_partition(args, graph, manifest)
    monitor = ResourceMonitor(args.output, int(args.memory_gb * GIB), int(args.output_gb * GIB), int(args.free_disk_gb * GIB))
    # Three int64 arrays plus NPZ headers and integration metadata; includes retained threshold graphs.
    monitor.check_output(24 * len(feature_ids) + 16 * 1024, "partition assignments/summary")
    _, sizes = write_assignments(args.output, feature_ids, supports, len(train_index), row_ids, graph_labels)
    summary = dict(
        status="complete", join_method=args.join_method,
        requested_threshold=args.threshold, **fitting,
        community_hard_edge_cap=args.community_max_edges, fixed_threshold=args.fixed_threshold,
        community_safety_fraction=COMMUNITY_SAFETY_FRACTION,
        features=int(len(feature_ids)), clusters=int(len(sizes)),
        singleton_clusters=int(np.sum(sizes == 1)), all_one_singletons=int(np.sum(supports == len(train_index))),
        graph_gib_before_pruning=directory_bytes(args.output) / GIB, edges_pruned=False, join=manifest,
    )
    (args.output / "integration.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
