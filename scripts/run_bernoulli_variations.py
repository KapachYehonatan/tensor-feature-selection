#!/usr/bin/env python3
"""Compare one Bernoulli-join variation at a time on cached real-data graphs."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.metrics import adjusted_rand_score  # noqa: E402

from research_utils import aggregate_columns, marginal_matrix, sqrt_sparse  # noqa: E402
from scripts.bernoulli_coproject import (  # noqa: E402
    VERSION as COPROJECTION_VERSION,
    canonical_labels,
    contract_graph,
    contracted_ids,
    initial_graph,
)
from scripts.run_bernoulli_correlation_join import (  # noqa: E402
    GRAPH_NAME, JOIN_PYTHON, aggregate_pair_clusters,
)
from scripts.run_method_comparison import (  # noqa: E402
    HOLDOUT_SEED, K_VALUES, MINIMUM_SAMPLES, MODEL_SEED, OUTPUT_ROOT,
    PREVALENCE_FRACTION, RSP_FDR, RSP_PERMUTATIONS, SCREEN_FDR,
    TEST_FRACTION, add_job_columns, add_job_values, bh_family,
    evaluate_ranked, feature_table, is_complete, permuted_targets,
    real_jobs, rsp_row, save_method, split_and_normalize,
)

ROOT = PROJECT / "outputs/bernoulli_variations"
WORKER = PROJECT / "scripts/bernoulli_variation_cluster_worker.py"
JOIN_COMMUNITY = Path("/home/kapachy/Microbium/bernoulli-correlation-join/community_detection.py")
COPROJECTIONS = {
    "coprojection_positive": "positive",
    "coprojection_absolute": "absolute",
}
VARIATIONS = (
    "max_variance", "bacterium_split", "signed_families", "kaminpar",
    *COPROJECTIONS,
)
LABELS = {
    "baseline": "Bernoulli join · mean",
    "max_variance": "Max-variance representative",
    "bacterium_split": "Bacterium-homogeneous",
    "signed_families": "Separate signed families",
    "kaminpar": "KaMinPar · matched K",
    "coprojection_positive": "Bernoulli contracted projection · positive",
    "coprojection_absolute": "Bernoulli contracted projection · absolute",
}
FINE_K = list(range(1, 11))
ALL_K = sorted(set(K_VALUES + FINE_K))
VERSION = 3
CLUSTER_VERSION = 3
WORKERS = min(36, int(os.environ.get("VARIATION_CPUS", "36")))
MAX_COPROJECTION_ROUNDS = 4
COPROJECTION_SOURCE = PROJECT / "scripts/bernoulli_coproject.py"


def graph_path(dataset, integration):
    root = OUTPUT_ROOT / "real" / dataset / GRAPH_NAME
    effective = float(integration["effective_threshold"])
    original = root / "graph"
    original_threshold = json.loads((original / "run.json").read_text())["threshold"]
    graph = original if float(original_threshold) == effective else root / f"graph-threshold-{effective:.2f}"
    if not (graph / "run.json").exists() or not (graph / "edges").exists():
        raise FileNotFoundError(f"stored exact graph is unavailable: {graph}")
    return graph


def aligned_labels(path, feature_ids):
    table = pd.read_csv(path, usecols=["feature_id", "cluster"])
    if table.feature_id.duplicated().any():
        raise ValueError("duplicate feature ID in clustering output")
    indexed = table.set_index("feature_id").reindex(feature_ids)
    if indexed.cluster.isna().any():
        raise ValueError("cluster assignments do not cover retained pair IDs")
    raw = indexed.cluster.to_numpy(dtype=np.int64, copy=True)
    if np.any(raw < 0):
        raw[raw < 0] = np.arange(int(raw.max()) + 1, int(raw.max()) + 1 + np.sum(raw < 0))
    _, labels = np.unique(raw, return_inverse=True)
    return labels.astype(np.int64)


def choose_representatives(train, labels, feature_ids):
    """Highest training variance of sqrt abundance, then lowest pair ID."""
    values = sqrt_sparse(train).tocsc()
    mean = np.asarray(values.mean(axis=0)).ravel()
    square = np.asarray(values.power(2).mean(axis=0)).ravel()
    variance = np.maximum(0, square - mean * mean)
    order = np.lexsort((feature_ids, -variance, labels))
    first = np.r_[True, labels[order[1:]] != labels[order[:-1]]]
    return order[first], variance


def split_by_bacterium(labels, bacteria):
    if len(labels) != len(bacteria):
        raise ValueError("bacteria and cluster labels are misaligned")
    keys = np.column_stack((labels, bacteria))
    groups, new_labels = np.unique(keys, axis=0, return_inverse=True)
    return groups, new_labels.astype(np.int64)


def strongest_negative_edges(graph, feature_ids, labels):
    """One internal edge per negative community; deterministic strength/ID ties."""
    ids = np.asarray(feature_ids, dtype=np.int64)
    order_ids = np.argsort(ids)
    sorted_ids = ids[order_ids]
    best = {}
    for path in sorted((graph / "edges").glob("part-*.npy")):
        shard = np.load(path, mmap_mode="r", allow_pickle=False)
        left = np.asarray(shard["feature_j"], dtype=np.int64)
        right = np.asarray(shard["feature_k"], dtype=np.int64)
        left_pos = np.searchsorted(sorted_ids, left)
        right_pos = np.searchsorted(sorted_ids, right)
        if (np.any(left_pos >= len(ids)) or np.any(right_pos >= len(ids))
                or np.any(sorted_ids[left_pos] != left) or np.any(sorted_ids[right_pos] != right)):
            raise ValueError("negative graph references a non-retained pair")
        groups = labels[order_ids[left_pos]]
        internal = groups == labels[order_ids[right_pos]]
        if not np.any(internal):
            continue
        groups = groups[internal]
        lo = np.minimum(left[internal], right[internal])
        hi = np.maximum(left[internal], right[internal])
        strength = np.abs(np.asarray(shard["phi"])[internal])
        chosen = np.lexsort((hi, lo, -strength, groups))
        chosen = chosen[np.r_[True, groups[chosen[1:]] != groups[chosen[:-1]]]]
        for index in chosen:
            key = int(groups[index])
            candidate = (-float(strength[index]), int(lo[index]), int(hi[index]))
            if key not in best or candidate < best[key]:
                best[key] = candidate
    return [(group, item[1], item[2], -item[0]) for group, item in sorted(best.items())]


def signed_features(train, test, feature_ids, positive, negative, negative_graph):
    positive_train, positive_test, sizes = aggregate_pair_clusters(train, test, positive)
    endpoints = strongest_negative_edges(negative_graph, feature_ids, negative)
    id_to_column = {int(value): row for row, value in enumerate(feature_ids)}
    rows, cols, values = [], [], []
    for column, (_, lo, hi, _) in enumerate(endpoints):
        rows.extend((id_to_column[lo], id_to_column[hi]))
        cols.extend((column, column))
        values.extend((1.0, -1.0))
    projection = sparse.csr_matrix((values, (rows, cols)), shape=(len(feature_ids), len(endpoints)))
    contrast_train = sqrt_sparse(train) @ projection
    contrast_test = sqrt_sparse(test) @ projection
    matrix_train = sparse.hstack((positive_train, contrast_train), format="csr")
    matrix_test = sparse.hstack((positive_test, contrast_test), format="csr")
    inventory = pd.DataFrame({
        "feature_id": np.arange(matrix_train.shape[1]),
        "family": ["positive_mean"] * positive_train.shape[1] + ["negative_contrast"] * len(endpoints),
        "community_id": list(range(len(sizes))) + [row[0] for row in endpoints],
        "cluster_size": list(sizes) + [int(np.sum(negative == row[0])) for row in endpoints],
        "endpoint_low": [np.nan] * len(sizes) + [row[1] for row in endpoints],
        "endpoint_high": [np.nan] * len(sizes) + [row[2] for row in endpoints],
        "edge_abs_phi": [np.nan] * len(sizes) + [row[3] for row in endpoints],
    })
    return matrix_train, matrix_test, inventory


def _projected_labels(graph, community, active_entities, log_path, threshold):
    """Run or reuse Leiden for one projected bacteria or gene graph."""
    graph_run = json.loads((graph / "run.json").read_text())
    settings = {
        "cluster_version": CLUSTER_VERSION,
        "graph_signature": graph_run["signature"],
        "objective": "modularity",
        "resolution": 1.0,
        "seed": 17_729,
        "leiden_iterations": -1,
        "threshold": float(threshold),
        "workers": WORKERS,
    }
    marker = community / "projection.json"
    valid = False
    if marker.exists() and (community / "clusters.csv.gz").exists():
        try:
            saved = json.loads(marker.read_text())
            valid = saved == settings
            if not valid and all(saved.get(key) == settings.get(key) for key in saved):
                summary = json.loads((community / "community.json").read_text())
                valid = all(summary.get(key) == settings[key] for key in (
                    "objective", "resolution", "seed", "leiden_iterations", "workers"))
        except (OSError, json.JSONDecodeError):
            pass
    if not valid:
        shutil.rmtree(community, ignore_errors=True)
        command = [
            str(JOIN_PYTHON), str(WORKER), "coprojection", str(graph),
            str(community), "--threshold", str(threshold),
            "--workers", str(WORKERS),
        ]
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            reason = log_path.read_text()[-3000:]
            raise RuntimeError(f"projected Leiden failed: {reason[-500:]}")
        marker.write_text(json.dumps(settings, indent=2, sort_keys=True) + "\n")
    labels = aligned_labels(community / "clusters.csv.gz", active_entities)
    return canonical_labels(active_entities, labels)


def _cluster_dimension(graph, root, pair_entity, feature_ids, sign_mode,
                       dimension, threshold, log_path):
    """Build once, then repeatedly contract one independent dimension."""
    active = np.unique(pair_entity[feature_ids]).astype(np.int64)
    original_nodes = active.copy()
    rows = []
    current_graph = root / f"round_01_{dimension}/graph"
    manifest = initial_graph(
        graph, current_graph, pair_entity, feature_ids, sign_mode, dimension)
    peak_rss_mb = 0.0
    converged = False

    for pass_number in range(1, MAX_COPROJECTION_ROUNDS + 1):
        stage = root / f"round_{pass_number:02d}_{dimension}"
        nodes = np.load(current_graph / "nodes.npy", mmap_mode="r", allow_pickle=False)
        node_ids = np.asarray(nodes["feature_id"], dtype=np.int64)
        labels = _projected_labels(
            current_graph, stage / "community", node_ids, log_path, threshold)
        community = json.loads((stage / "community/community.json").read_text())
        new_ids = contracted_ids(node_ids, labels)
        merged = len(np.unique(new_ids)) < len(node_ids)
        lookup = np.full(int(node_ids.max()) + 1, -1, dtype=np.int64)
        lookup[node_ids] = new_ids
        previous_labels = np.unique(original_nodes, return_inverse=True)[1]
        original_nodes = lookup[original_nodes]
        current_labels = np.unique(original_nodes, return_inverse=True)[1]
        rows.append({
            "dimension": dimension, "pass": pass_number,
            "input_nodes": len(node_ids), "input_edges": manifest["edge_count"],
            "output_clusters": int(np.unique(new_ids).size), "merged": merged,
            "partition_ari": adjusted_rand_score(previous_labels, current_labels),
            "modularity": community.get("modularity"),
            "community_seconds": community.get("wall_s"),
        })
        peak_rss_mb = max(peak_rss_mb, community.get("peak_rss_bytes", 0) / 1024**2)
        if not merged:
            converged = True
            break
        if pass_number == MAX_COPROJECTION_ROUNDS:
            break
        next_graph = root / f"round_{pass_number + 1:02d}_{dimension}/graph"
        next_manifest = contract_graph(current_graph, next_graph, new_ids)
        if next_manifest["vertices"] >= manifest["vertices"]:
            raise RuntimeError("a merging contraction did not reduce the node count")
        shutil.rmtree(current_graph / "edges", ignore_errors=True)
        current_graph, manifest = next_graph, next_manifest

    if rows[-1]["merged"]:
        final_graph = root / f"final_{dimension}/graph"
        final_manifest = contract_graph(current_graph, final_graph, new_ids)
        shutil.rmtree(current_graph / "edges", ignore_errors=True)
        current_graph, manifest = final_graph, final_manifest

    return active, current_labels.astype(np.int64), rows, {
        "passes": len(rows), "converged": converged,
        "clusters": int(np.unique(current_labels).size),
        "singleton_fraction": float(np.mean(np.bincount(current_labels) == 1)),
        "final_nodes": manifest["vertices"], "final_edges": manifest["edge_count"],
        "peak_rss_mb": peak_rss_mb,
    }


def coprojection_clusters(graph, destination, pair_bacteria, pair_gene,
                          feature_ids, sign_mode, signature, threshold):
    """Cluster bacteria and genes as independent contracted projections."""
    root = destination / "coprojection"
    marker = root / "complete.json"
    if marker.exists():
        saved = json.loads(marker.read_text())
        assignments = root / "assignments.npz"
        if (saved.get("signature") == signature
                and saved.get("architecture_version") == COPROJECTION_VERSION
                and assignments.exists()):
            with np.load(assignments) as values:
                return (
                    values["active_bacteria"], values["bacteria_labels"],
                    values["active_genes"], values["gene_labels"], saved,
                )
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)
    log_path = destination / "clustering.log"
    log_path.unlink(missing_ok=True)
    started = perf_counter()
    active_bacteria, bacteria_labels, bacteria_rows, bacteria_summary = _cluster_dimension(
        graph, root, pair_bacteria, feature_ids, sign_mode,
        "bacteria", threshold, log_path)
    active_genes, gene_labels, gene_rows, gene_summary = _cluster_dimension(
        graph, root, pair_gene, feature_ids, sign_mode,
        "gene", threshold, log_path)
    np.savez_compressed(
        root / "assignments.npz", active_bacteria=active_bacteria,
        bacteria_labels=bacteria_labels, active_genes=active_genes,
        gene_labels=gene_labels)
    bacteria_sizes, gene_sizes = np.bincount(bacteria_labels), np.bincount(gene_labels)
    pd.DataFrame({
        "bacterium_id": active_bacteria, "cluster": bacteria_labels,
        "cluster_size": bacteria_sizes[bacteria_labels],
    }).to_csv(root / "bacteria_assignments.csv.gz", index=False, compression="gzip")
    pd.DataFrame({
        "gene_id": active_genes, "cluster": gene_labels,
        "cluster_size": gene_sizes[gene_labels],
    }).to_csv(root / "gene_assignments.csv.gz", index=False, compression="gzip")
    pd.DataFrame(bacteria_rows + gene_rows).to_csv(root / "iterations.csv", index=False)
    summary = {
        "signature": signature, "status": "complete",
        "architecture_version": COPROJECTION_VERSION, "sign_mode": sign_mode,
        "rounds": max(bacteria_summary["passes"], gene_summary["passes"]),
        "converged": bacteria_summary["converged"] and gene_summary["converged"],
        "bacteria_passes": bacteria_summary["passes"],
        "gene_passes": gene_summary["passes"],
        "bacteria_clusters": len(bacteria_sizes), "gene_clusters": len(gene_sizes),
        "bacteria_singleton_fraction": bacteria_summary["singleton_fraction"],
        "gene_singleton_fraction": gene_summary["singleton_fraction"],
        "final_bacteria_edges": bacteria_summary["final_edges"],
        "final_gene_edges": gene_summary["final_edges"],
        "runtime_seconds": perf_counter() - started,
        "peak_rss_mb": max(bacteria_summary["peak_rss_mb"], gene_summary["peak_rss_mb"]),
    }
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(marker)
    return active_bacteria, bacteria_labels, active_genes, gene_labels, summary


def _clustered_marginals(train, test, pair_entity, active_entities, labels):
    total = int(pair_entity.max()) + 1
    train_marginal = marginal_matrix(train, pair_entity, total)[:, active_entities]
    test_marginal = marginal_matrix(test, pair_entity, total)[:, active_entities]
    train_cluster, compact = aggregate_columns(train_marginal, labels)
    test_cluster, test_compact = aggregate_columns(test_marginal, labels)
    if not np.array_equal(compact, test_compact):
        raise RuntimeError("train and test cluster columns changed")
    return sqrt_sparse(train_cluster), sqrt_sparse(test_cluster), np.bincount(compact)


def evaluate_coprojection(job, variation, destination, signature, train, test,
                          y_train, y_test, retained_bacteria, retained_genes,
                          active_bacteria, bacteria_labels, active_genes,
                          gene_labels, cluster_summary, graph, threshold, started):
    """BH-test the two final marginal families and save standard outputs."""
    bacteria_train, bacteria_test, bacteria_sizes = _clustered_marginals(
        train, test, retained_bacteria, active_bacteria, bacteria_labels)
    gene_train, gene_test, gene_sizes = _clustered_marginals(
        train, test, retained_genes, active_genes, gene_labels)
    targets = permuted_targets(y_train, f"real:{job['dataset']}:None")
    bacteria_bh = bh_family(bacteria_train, targets)
    gene_bh = bh_family(gene_train, targets)
    levels = np.concatenate([
        np.full(bacteria_train.shape[1], "bacterium"),
        np.full(gene_train.shape[1], "gene"),
    ])
    ids = np.concatenate([
        np.arange(bacteria_train.shape[1]), np.arange(gene_train.shape[1])])
    scores = {key: np.concatenate([bacteria_bh[key], gene_bh[key]])
              for key in ("p", "q", "effect")}
    type_order = np.concatenate([
        np.zeros(bacteria_train.shape[1], dtype=np.int8),
        np.ones(gene_train.shape[1], dtype=np.int8)])
    order = np.lexsort((ids, type_order, scores["p"], scores["q"]))
    selected_order = order[scores["q"][order] <= SCREEN_FDR]
    selected = feature_table(selected_order, levels, ids, scores)
    sizes = np.concatenate([bacteria_sizes, gene_sizes])
    selected["cluster_size"] = sizes[selected_order]
    ranking = selected.iloc[:max(ALL_K)].copy()
    combined_train = sparse.hstack((bacteria_train, gene_train), format="csr")
    combined_test = sparse.hstack((bacteria_test, gene_test), format="csr")
    ranked_train = combined_train[:, selected_order[:max(ALL_K)]]
    ranked_test = combined_test[:, selected_order[:max(ALL_K)]]
    output_counts = {
        "bacteria": bacteria_train.shape[1], "gene": gene_train.shape[1],
        "pair": 0, "pair_cluster": 0,
    }
    selected_counts = {
        "bacteria": int(np.sum(levels[selected_order] == "bacterium")),
        "gene": int(np.sum(levels[selected_order] == "gene")),
        "pair": 0, "pair_cluster": 0,
    }
    metrics, predictions = evaluate_ranked(
        variation, ranked_train, ranked_test, ranking, y_train, y_test,
        output_counts, selected_counts, ALL_K, LABELS[variation])
    add_job_columns(metrics, job)
    add_job_columns(predictions, job)
    discoveries = (np.asarray(bacteria_bh["discoveries"])
                   + np.asarray(gene_bh["discoveries"]))
    rsp = add_job_values(rsp_row(variation, discoveries), job)
    inventory = pd.DataFrame({
        "level": levels, "feature_id": ids, "cluster_size": sizes})
    inventory.to_csv(
        destination / "cluster_inventory.csv.gz", index=False, compression="gzip")
    peak_rss_mb = max(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        cluster_summary["peak_rss_mb"],
    )
    save_method(
        destination, signature, ranking, ranked_train, ranked_test,
        metrics, predictions, rsp, selected, perf_counter() - started,
        {
            "clusters": int(len(bacteria_sizes) + len(gene_sizes)),
            "architecture_version": COPROJECTION_VERSION,
            "bacteria_clusters": int(len(bacteria_sizes)),
            "gene_clusters": int(len(gene_sizes)),
            "candidate_features": int(len(sizes)),
            "selected_features": len(selected_order),
            "selected_bacteria": selected_counts["bacteria"],
            "selected_genes": selected_counts["gene"],
            "effective_threshold": threshold, "graph": str(graph),
            "clustering_runtime_seconds": cluster_summary["runtime_seconds"],
            "clustering_peak_rss_mb": cluster_summary["peak_rss_mb"],
            "peak_rss_mb": peak_rss_mb,
            "rounds": cluster_summary["rounds"],
            "converged": cluster_summary["converged"],
        })
    (destination / "resource_limit.json").unlink(missing_ok=True)


def fingerprint(job, y, feature_ids, variation, graph):
    source = Path(job["input"])
    baseline = OUTPUT_ROOT / "real" / job["dataset"] / GRAPH_NAME
    files = [source / "X_valid.npz", source / "samples.csv", source / "edges.npz",
             baseline / "assignments.npz", baseline / "integration.json", graph / "run.json",
             Path(__file__)]
    if variation in {"signed_families", "kaminpar", *COPROJECTIONS}:
        files.extend((WORKER, JOIN_COMMUNITY))
    if variation in COPROJECTIONS:
        files.append(COPROJECTION_SOURCE)
    settings = {
        "version": VERSION, "variation": variation,
        "files": [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files],
        "k": ALL_K, "split": [TEST_FRACTION, HOLDOUT_SEED],
        "prevalence": [PREVALENCE_FRACTION, MINIMUM_SAMPLES],
        "screen_fdr": SCREEN_FDR, "rsp_fdr": RSP_FDR,
        "permutations": RSP_PERMUTATIONS, "model_seed": MODEL_SEED,
        "workers": WORKERS if variation in {"signed_families", "kaminpar", *COPROJECTIONS} else None,
        "coprojection_rounds": MAX_COPROJECTION_ROUNDS if variation in COPROJECTIONS else None,
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(np.asarray(y, dtype=np.uint8).tobytes())
    digest.update(np.asarray(feature_ids, dtype=np.int64).tobytes())
    return digest.hexdigest()


def run_one(job, variation):
    started = perf_counter()
    source = Path(job["input"])
    matrix = sparse.load_npz(source / "X_valid.npz").tocsr()
    y = pd.read_csv(source / "samples.csv")["y"].to_numpy(dtype=int)
    train_idx, test_idx, mask, _, _, train, test = split_and_normalize(matrix, y)
    ids = np.flatnonzero(mask).astype(np.int64)
    baseline = OUTPUT_ROOT / "real" / job["dataset"] / GRAPH_NAME
    baseline_completion = OUTPUT_ROOT / "real" / job["dataset"] / "bernoulli_correlation_join/complete.json"
    if any((source / name).stat().st_mtime_ns > baseline_completion.stat().st_mtime_ns
           for name in ("X_valid.npz", "samples.csv", "edges.npz")):
        raise RuntimeError("preprocessed input is newer than the completed baseline graph")
    split_matches = False
    for split_file in (baseline.parent / f"{GRAPH_NAME}_split.npz",
                       baseline.parent / "bernoulli_join_split.npz"):
        if split_file.exists():
            with np.load(split_file) as saved:
                split_matches |= (np.array_equal(saved["train_index"], train_idx)
                                  and np.array_equal(saved["test_index"], test_idx)
                                  and np.array_equal(saved["feature_ids"], ids))
    if not split_matches:
        raise ValueError("stored graph split does not match the comparison holdout")
    with np.load(baseline / "assignments.npz") as saved:
        if not np.array_equal(saved["feature_ids"], ids):
            raise ValueError("baseline graph was not built on this training split")
        baseline_labels = saved["clusters"].astype(np.int64)
    integration = json.loads((baseline / "integration.json").read_text())
    graph = graph_path(job["dataset"], integration)
    signature = fingerprint(job, y, ids, variation, graph)
    destination = ROOT / job["dataset"] / variation
    completed = is_complete(destination, signature)
    if completed and variation in COPROJECTIONS:
        try:
            completed = (
                json.loads((destination / "complete.json").read_text()).get(
                    "architecture_version"
                ) == COPROJECTION_VERSION
            )
        except (OSError, json.JSONDecodeError):
            completed = False
    if completed:
        print(f"Cached {job['dataset']} {variation}", flush=True)
        return
    if variation == "max_variance":
        representatives, variance = choose_representatives(train, baseline_labels, ids)
        x_train = sqrt_sparse(train[:, representatives])
        x_test = sqrt_sparse(test[:, representatives])
        inventory = pd.DataFrame({
            "feature_id": np.arange(len(representatives)), "family": "representative",
            "representative_pair_id": ids[representatives],
            "training_variance": variance[representatives],
            "cluster_size": np.bincount(baseline_labels)[baseline_labels[representatives]],
        })
        assignments = baseline_labels
    elif variation == "bacterium_split":
        with np.load(source / "edges.npz") as saved:
            bacteria = saved["pair_bacteria"][mask]
        groups, assignments = split_by_bacterium(baseline_labels, bacteria)
        x_train, x_test, sizes = aggregate_pair_clusters(train, test, assignments)
        inventory = pd.DataFrame({
            "feature_id": np.arange(len(groups)), "family": "bacterium_mean",
            "baseline_cluster": groups[:, 0], "bacterium_id": groups[:, 1],
            "cluster_size": sizes,
        })
    elif variation in COPROJECTIONS:
        with np.load(source / "edges.npz") as saved:
            pair_bacteria = saved["pair_bacteria"].astype(np.int64)
            pair_gene = saved["pair_gene"].astype(np.int64)
        try:
            active_bacteria, bacteria_labels, active_genes, gene_labels, summary = (
                coprojection_clusters(
                    graph, destination, pair_bacteria, pair_gene, ids,
                    COPROJECTIONS[variation], signature,
                    float(integration["effective_threshold"])))
        except Exception as error:
            destination.mkdir(parents=True, exist_ok=True)
            message = str(error)
            limited = any(token in message.lower() for token in (
                "memory", "reserve", "limit", "exceed", "killed", "disk", "space"))
            (destination / "resource_limit.json").write_text(json.dumps({
                "status": "resource_limited" if limited else "failed",
                "reason": message, "graph": str(graph), "variation": variation,
            }, indent=2) + "\n")
            raise
        evaluate_coprojection(
            job, variation, destination, signature, train, test,
            y[train_idx], y[test_idx], pair_bacteria[mask], pair_gene[mask],
            active_bacteria, bacteria_labels, active_genes, gene_labels,
            summary, graph, float(integration["effective_threshold"]), started)
        print(
            f"Completed {job['dataset']} {variation}: "
            f"{summary['bacteria_clusters']:,} bacteria clusters, "
            f"{summary['gene_clusters']:,} gene clusters, "
            f"{perf_counter() - started:.1f}s", flush=True)
        return
    else:
        community = destination / "community"
        cluster_settings = {
            "version": CLUSTER_VERSION, "variation": variation,
            "graph": str(graph.resolve()),
            "graph_mtime_ns": (graph / "run.json").stat().st_mtime_ns,
            "threshold": float(integration["effective_threshold"]),
            "workers": WORKERS,
            "signed_leiden_iterations": 2 if variation == "signed_families" else None,
        }
        cluster_marker = community / "variation.json"
        cluster_valid = False
        if cluster_marker.exists():
            try:
                cluster_valid = json.loads(cluster_marker.read_text()) == cluster_settings
            except (OSError, json.JSONDecodeError):
                pass
        if not cluster_valid and community.exists():
            shutil.rmtree(community)
            (destination / "clustering_runtime.json").unlink(missing_ok=True)
        cluster_file = (community / "community/clusters.csv.gz" if variation == "kaminpar"
                        else community / "positive_community/clusters.csv.gz")
        fresh_clustering = not cluster_file.exists()
        if fresh_clustering:
            clustering_started = perf_counter()
            edge_communities = int(integration["community"]["clusters"]
                                   - integration["community"]["eligible_vertices"]
                                   + integration["community"]["edge_vertices"])
            if variation == "kaminpar" and edge_communities < 1:
                raise ValueError("no edge-bearing baseline communities for KaMinPar")
            command = [str(JOIN_PYTHON), str(WORKER), variation, str(graph), str(community),
                       "--threshold", str(integration["effective_threshold"]),
                       "--workers", str(WORKERS)]
            if variation == "kaminpar":
                command += ["--k", str(edge_communities)]
            destination.mkdir(parents=True, exist_ok=True)
            with (destination / "clustering.log").open("a", encoding="utf-8") as log:
                result = subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                reason = (destination / "clustering.log").read_text()[-3000:]
                limited = any(token in reason.lower() for token in (
                    "resourcelimit", "memory", "reserve", "overflow", "exceed",
                    "killed", "no space", "disk", "budget"
                ))
                (destination / "resource_limit.json").write_text(json.dumps({
                    "status": "resource_limited" if limited else "failed",
                    "returncode": result.returncode,
                    "reason": reason, "graph": str(graph), "variation": variation,
                }, indent=2) + "\n")
                raise RuntimeError(f"{variation} clustering failed for {job['dataset']}: {reason[-400:]}")
            cluster_marker.write_text(json.dumps(cluster_settings, indent=2, sort_keys=True) + "\n")
            (destination / "clustering_runtime.json").write_text(json.dumps({
                "wall_seconds": perf_counter() - clustering_started,
                "includes_graph_filtering": variation == "signed_families",
            }, indent=2) + "\n")
        assignments = aligned_labels(cluster_file, ids)
        if variation == "kaminpar":
            x_train, x_test, sizes = aggregate_pair_clusters(train, test, assignments)
            inventory = pd.DataFrame({"feature_id": np.arange(len(sizes)),
                                      "family": "kaminpar_mean", "cluster_size": sizes})
        else:
            negative_graph = community / "negative_graph"
            negative = aligned_labels(community / "negative_community/clusters.csv.gz", ids)
            x_train, x_test, inventory = signed_features(
                train, test, ids, assignments, negative, negative_graph)
            np.savez_compressed(destination / "signed_membership.npz",
                                feature_ids=ids, positive=assignments, negative=negative)
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "assignments.npz", feature_ids=ids, clusters=assignments)
    inventory.to_csv(destination / "inventory.csv.gz", index=False, compression="gzip")
    targets = permuted_targets(y[train_idx], f"real:{job['dataset']}:None")
    scores = bh_family(x_train, targets)
    selected_order = scores["order"][scores["q"][scores["order"]] <= SCREEN_FDR]
    levels = np.full(x_train.shape[1], "pair_cluster")
    columns = np.arange(x_train.shape[1])
    ranking = feature_table(selected_order, levels, columns, scores, max(ALL_K))
    selected = feature_table(selected_order, levels, columns, scores)
    selected = selected.merge(inventory, on="feature_id", how="left", validate="one_to_one")
    top_train, top_test = x_train[:, selected_order[:max(ALL_K)]], x_test[:, selected_order[:max(ALL_K)]]
    counts = {"bacteria": 0, "gene": 0, "pair": 0, "pair_cluster": x_train.shape[1]}
    selected_counts = {**counts, "pair_cluster": len(selected_order)}
    metrics, predictions = evaluate_ranked(
        variation, top_train, top_test, ranking, y[train_idx], y[test_idx],
        counts, selected_counts, ALL_K, LABELS[variation],
    )
    add_job_columns(metrics, job)
    add_job_columns(predictions, job)
    rsp = add_job_values(rsp_row(variation, scores["discoveries"]), job)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    if variation in {"signed_families", "kaminpar"}:
        names = (("positive_community", "negative_community") if variation == "signed_families"
                 else ("community",))
        for name in names:
            summary = json.loads((destination / "community" / name / "community.json").read_text())
            peak_mb = max(peak_mb, summary["peak_rss_bytes"] / 1024**2)
        runtime_file = destination / "clustering_runtime.json"
        clustering_seconds = (json.loads(runtime_file.read_text())["wall_seconds"]
                              if runtime_file.exists() else sum(
                                  json.loads((destination / "community" / name / "community.json").read_text())["wall_s"]
                                  for name in names))
    else:
        clustering_seconds = 0.0
    elapsed = perf_counter() - started
    if variation in {"signed_families", "kaminpar"} and not fresh_clustering:
        elapsed += clustering_seconds
    selected_families = selected["family"].value_counts()
    save_method(destination, signature, ranking, top_train, top_test,
                metrics, predictions, rsp, selected, elapsed,
                {"clusters": int(np.unique(assignments).size),
                 "candidate_features": int(x_train.shape[1]),
                 "selected_features": len(selected_order),
                 "selected_positive_mean": int(selected_families.get("positive_mean", 0)),
                 "selected_negative_contrast": int(selected_families.get("negative_contrast", 0)),
                 "effective_threshold": float(integration["effective_threshold"]),
                 "graph": str(graph),
                 "clustering_runtime_seconds": clustering_seconds,
                 "peak_rss_mb": peak_mb})
    print(f"Completed {job['dataset']} {variation}: {x_train.shape[1]:,} candidates, "
          f"{len(selected_order):,} selected, {elapsed:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variation", choices=VARIATIONS, required=True)
    parser.add_argument("--dataset", choices=[job["dataset"] for job in real_jobs()] + ["all"],
                        default="RubelMA_2020")
    args = parser.parse_args()
    jobs = real_jobs()
    if args.dataset != "all":
        jobs = [job for job in jobs if job["dataset"] == args.dataset]
    for job in jobs:
        try:
            run_one(job, args.variation)
        except Exception as error:
            print(f"FAILED {job['dataset']} {args.variation}: {error}", flush=True)
            limit = ROOT / job["dataset"] / args.variation / "resource_limit.json"
            if (args.dataset == "all" and limit.exists()
                    and json.loads(limit.read_text()).get("status") == "resource_limited"):
                print(f"Resource-limited: continuing to the next dataset", flush=True)
                continue
            raise
    from scripts.summarize_bernoulli_variations import summarize
    summarize()


if __name__ == "__main__":
    main()
