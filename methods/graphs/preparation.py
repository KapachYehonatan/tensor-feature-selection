"""Pack sample-by-pair values and build/filter exact Bernoulli graphs."""
from pathlib import Path
import json
import os
import shutil
import numpy as np
from scipy import sparse
from methods.graphs.resources import GIB

CHUNK_FEATURES = 10_000

def pack_sparse_detection(matrix, feature_ids, output):
    """Pack sample×pair sparse values to pair×ceil(samples/64) words; return pair supports."""
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


def build_graph(args, train_index, feature_ids):
    """Read args.matrix/split rows, build exact edges at args.output, return manifest/supports."""
    from methods.graphs.exact import run_join, load_packed_dataset, packed_supports
    packed, graph = args.output / "packed", args.output / "graph"
    run_path = graph / "run.json"
    if args.output.exists():
        if not run_path.exists():
            raise RuntimeError("existing output has no completed exact-join manifest")
        manifest = json.loads(run_path.read_text(encoding="utf-8"))
        expected_method = "exact"
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
        manifest = run_join(
            packed, graph, args.threshold, args.memory_gb, args.output_gb,
            args.free_disk_gb, args.workers, args.working_memory_gb,
            min_one_support=1,
        )
    if manifest["status"] != "complete":
        raise RuntimeError(f"exact join stopped: {manifest['status']}: {manifest.get('reason', '')}")

    return manifest, supports

