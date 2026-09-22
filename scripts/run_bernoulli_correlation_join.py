#!/usr/bin/env python3
"""Run one exact detection-join comparison method over real datasets."""

from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import sys
from time import perf_counter

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

from research_utils import sqrt_sparse  # noqa: E402
from scripts.run_method_comparison import (  # noqa: E402
    CACHE_VERSION, HOLDOUT_SEED, K_VALUES, METHOD_LABELS, MINIMUM_SAMPLES,
    MODEL_SEED, OUTPUT_ROOT, PREVALENCE_FRACTION, RSP_BETA, RSP_FDR,
    RSP_PERMUTATIONS, RSP_SEED, SCREEN_FDR, TEST_FRACTION,
    add_job_columns, add_job_values, bh_family, evaluate_ranked, feature_table,
    is_complete, permuted_targets, real_jobs, rsp_row, save_method,
    split_and_normalize,
)


JOIN_METHOD = os.environ.get("JOIN_METHOD", "bernoulli").strip().lower()
if JOIN_METHOD not in {"bernoulli", "jaccard"}:
    raise ValueError("JOIN_METHOD must be bernoulli or jaccard")
METHOD = (
    "bernoulli_correlation_join" if JOIN_METHOD == "bernoulli" else "jaccard_join"
)
GRAPH_NAME = (
    "bernoulli_join_graph" if JOIN_METHOD == "bernoulli" else "jaccard_join_graph"
)
THRESHOLD = 0.7
JOIN_PYTHON = Path("/home/kapachy/Microbium/bernoulli-correlation-join/.venv/bin/python")
JOIN_SOURCE = Path("/home/kapachy/Microbium/bernoulli-correlation-join")
WORKER = PROJECT / "scripts/bernoulli_join_worker.py"
WORKERS = 36
MEMORY_GIB = 125
WORKING_MEMORY_GIB = 40
COMMUNITY_WORKING_MEMORY_GIB = 125
OUTPUT_GIB = 40
FREE_DISK_GIB = 500
MEMORY_RESERVE_GIB = 64
KEEP_GRAPH_GIB = 20
SIDE_VERSION = 3


def aggregate_pair_clusters(train, test, labels):
    """Average normalized original pair abundances within learned clusters."""
    labels = np.asarray(labels, dtype=int)
    if labels.shape != (train.shape[1],) or train.shape[1] != test.shape[1]:
        raise ValueError("cluster labels must align with train and test columns")
    if np.any(labels < 0):
        raise ValueError("every retained feature must have a nonnegative cluster")
    cluster_count = int(labels.max()) + 1
    sizes = np.bincount(labels, minlength=cluster_count)
    assignment = sparse.csr_matrix(
        (1.0 / sizes[labels], (np.arange(len(labels)), labels)),
        shape=(len(labels), cluster_count),
    )
    return sqrt_sparse(train @ assignment), sqrt_sparse(test @ assignment), sizes


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def signature(job, y, graph_only=False):
    source = Path(job["input"])
    files = [source / "X_valid.npz", source / "samples.csv"]
    settings = {
        "version": SIDE_VERSION, "cache_version": CACHE_VERSION,
        "method": METHOD, "threshold": THRESHOLD,
        "split": [TEST_FRACTION, HOLDOUT_SEED],
        "prevalence": [PREVALENCE_FRACTION, MINIMUM_SAMPLES],
        "inputs": [(str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in files],
        "join_sources": {
            name: file_hash(JOIN_SOURCE / name)
            for name in (
                "bernoulli_join.py", "community_detection.py",
                *(("baseline_graphs.py",) if JOIN_METHOD == "jaccard" else ()),
            )
        },
        "worker": file_hash(WORKER),
    }
    if not graph_only:
        settings.update({
            "k": K_VALUES, "screen_fdr": SCREEN_FDR, "model_seed": MODEL_SEED,
            "rsp": [RSP_FDR, RSP_BETA, RSP_PERMUTATIONS, RSP_SEED],
        })
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(np.asarray(y, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def run_dataset(job):
    started = perf_counter()
    source = Path(job["input"])
    matrix = sparse.load_npz(source / "X_valid.npz").tocsr()
    y = pd.read_csv(source / "samples.csv")["y"].to_numpy(dtype=int)
    if matrix.shape[0] != len(y) or set(np.unique(y)) != {0, 1}:
        raise ValueError(f"unaligned matrix or invalid target for {job['dataset']}")
    run_signature = signature(job, y)
    graph_signature = signature(job, y, graph_only=True)
    destination = OUTPUT_ROOT / "real" / job["dataset"]
    destination.mkdir(parents=True, exist_ok=True)
    method_path = destination / METHOD
    cluster_path = destination / GRAPH_NAME
    if is_complete(method_path, run_signature) and (cluster_path / "assignments.npz").exists():
        print(f"Cached: {job['dataset']}", flush=True)
        return
    train_index, test_index, mask, support, prevalence_threshold, train, test = split_and_normalize(matrix, y)
    feature_ids = np.flatnonzero(mask).astype(np.int64)
    if not len(feature_ids):
        raise RuntimeError("prevalence filtering retained no pair features")
    if np.any(support[mask] == 0):
        raise RuntimeError("prevalence filtering retained an all-zero feature")
    y_train, y_test = y[train_index], y[test_index]
    # Preserve completed join results after verifying current input dates and
    # retained feature IDs. Worker-only fixes must not rerun finished Leiden jobs.
    old_completion = method_path / "complete.json"
    old_integration = cluster_path / "integration.json"
    assignment_path = cluster_path / "assignments.npz"
    if old_completion.exists() and old_integration.exists():
        completion = json.loads(old_completion.read_text(encoding="utf-8"))
        integration = json.loads(old_integration.read_text(encoding="utf-8"))
        required = (
            "metrics.csv", "predictions.csv.gz", "ranked_features.csv.gz",
            "ranked_train.npz", "ranked_test.npz", "selected_features.csv.gz",
        )
        input_is_older = all(
            path.stat().st_mtime_ns <= old_completion.stat().st_mtime_ns
            for path in (source / "X_valid.npz", source / "samples.csv")
        )
        cached_ids = np.array([], dtype=int)
        if assignment_path.exists():
            with np.load(assignment_path) as saved:
                cached_ids = saved["feature_ids"]
        completed_threshold = completion.get(
            "requested_similarity_threshold",
            completion.get("requested_correlation_threshold"),
        )
        completed_method = integration.get("join_method", "bernoulli")
        if (
            completed_threshold == THRESHOLD
            and integration.get("requested_threshold") == THRESHOLD
            and completed_method == JOIN_METHOD
            and input_is_older
            and np.array_equal(cached_ids, feature_ids)
            and all((method_path / name).exists() for name in required)
        ):
            completion["signature"] = run_signature
            old_completion.write_text(
                json.dumps(completion, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            print(f"Cached compatible result: {job['dataset']}", flush=True)
            return
    cluster_marker = destination / f"{GRAPH_NAME}_complete.json"
    cluster_ok = False
    if cluster_marker.exists() and (cluster_path / "assignments.npz").exists():
        try:
            cluster_ok = (
                json.loads(cluster_marker.read_text(encoding="utf-8")).get("signature")
                == graph_signature
            )
        except (OSError, json.JSONDecodeError):
            cluster_ok = False
    if not cluster_ok:
        resumable = False
        run_path = cluster_path / "graph" / "run.json"
        if (
            run_path.exists()
            and (cluster_path / "packed" / "metadata.json").exists()
            and (cluster_path / "graph" / "edges").exists()
        ):
            try:
                run = json.loads(run_path.read_text(encoding="utf-8"))
                resumable = (
                    run.get("status") == "complete"
                    and run.get("threshold", 1.0) <= THRESHOLD
                )
            except (OSError, json.JSONDecodeError):
                pass
        if not resumable:
            shutil.rmtree(cluster_path, ignore_errors=True)
        cluster_marker.unlink(missing_ok=True)
        split_path = destination / f"{GRAPH_NAME}_split.npz"
        np.savez_compressed(
            split_path, train_index=train_index, test_index=test_index,
            feature_ids=feature_ids, prevalence_threshold=prevalence_threshold,
        )
        command = [
            str(JOIN_PYTHON), str(WORKER), str(source / "X_valid.npz"),
            str(split_path), str(cluster_path), "--threshold", str(THRESHOLD),
            "--join-method", JOIN_METHOD,
            "--workers", str(WORKERS), "--memory-gb", str(MEMORY_GIB),
            "--working-memory-gb", str(WORKING_MEMORY_GIB),
            "--community-working-memory-gb", str(COMMUNITY_WORKING_MEMORY_GIB),
            "--output-gb", str(OUTPUT_GIB), "--free-disk-gb", str(FREE_DISK_GIB),
            "--memory-reserve-gb", str(MEMORY_RESERVE_GIB),
            "--keep-graph-gb", str(KEEP_GRAPH_GIB),
        ]
        log_path = PROJECT / "logs" / f"{JOIN_METHOD}_join_{job['dataset']}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Exact join log: {log_path}", flush=True)
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(
                command, cwd=PROJECT, check=True, stdout=log,
                stderr=subprocess.STDOUT,
            )
        temporary_marker = cluster_marker.with_suffix(".json.tmp")
        temporary_marker.write_text(
            json.dumps({"cache_version": CACHE_VERSION, "signature": graph_signature}, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary_marker.replace(cluster_marker)

    with np.load(cluster_path / "assignments.npz") as saved:
        assigned_ids = saved["feature_ids"]
        labels = saved["clusters"]
    if not np.array_equal(assigned_ids, feature_ids):
        raise RuntimeError(f"cached {JOIN_METHOD} assignments do not match retained features")
    integration = json.loads((cluster_path / "integration.json").read_text(encoding="utf-8"))
    effective_threshold = float(integration["effective_threshold"])
    cluster_train, cluster_test, sizes = aggregate_pair_clusters(train, test, labels)
    targets = permuted_targets(y_train, f"real:{job['dataset']}:None")
    result = bh_family(cluster_train, targets)
    selected_order = result["order"][result["q"][result["order"]] <= SCREEN_FDR]
    cluster_ids = np.arange(cluster_train.shape[1])
    levels = np.full(cluster_train.shape[1], "pair_cluster")
    ranking = feature_table(selected_order, levels, cluster_ids, result, max(K_VALUES))
    selected = feature_table(selected_order, levels, cluster_ids, result)
    selected["cluster_size"] = sizes[selected["feature_id"].to_numpy(dtype=int)]
    selected["similarity"] = JOIN_METHOD
    selected["requested_similarity_threshold"] = THRESHOLD
    selected["effective_similarity_threshold"] = effective_threshold
    ranked_train = cluster_train[:, selected_order[:max(K_VALUES)]]
    ranked_test = cluster_test[:, selected_order[:max(K_VALUES)]]
    output_counts = {
        "bacteria": 0, "gene": 0, "pair": 0,
        "pair_cluster": cluster_train.shape[1],
    }
    selected_counts = {
        "bacteria": 0, "gene": 0, "pair": 0,
        "pair_cluster": len(selected_order),
    }
    metrics, predictions = evaluate_ranked(
        METHOD, ranked_train, ranked_test, ranking, y_train, y_test,
        output_counts, selected_counts,
    )
    add_job_columns(metrics, job)
    add_job_columns(predictions, job)
    for table in (metrics, predictions):
        table["similarity"] = JOIN_METHOD
        table["requested_similarity_threshold"] = THRESHOLD
        table["effective_similarity_threshold"] = effective_threshold
    rsp = add_job_values(rsp_row(METHOD, result["discoveries"]), job)
    elapsed = perf_counter() - started
    save_method(
        method_path, run_signature, ranking, ranked_train, ranked_test,
        metrics, predictions, rsp, selected, elapsed,
        completion_extra={
            "join_runtime_seconds": integration["join"]["wall_s"],
            "join_peak_rss_mb": integration["join"]["peak_rss_bytes"] / 1024**2,
            "clusters": int(len(sizes)), "retained_pairs": int(len(feature_ids)),
            "similarity": JOIN_METHOD,
            "requested_similarity_threshold": THRESHOLD,
            "effective_similarity_threshold": effective_threshold,
        },
    )
    print(
        f"Completed {job['dataset']}: samples={len(y)}, retained={len(feature_ids):,}, "
        f"clusters={len(sizes):,}, selected={len(selected_order):,}, "
        f"prevalence_support={prevalence_threshold}, "
        f"{JOIN_METHOD}_threshold={effective_threshold:.2f}, elapsed={elapsed:.1f}s",
        flush=True,
    )


def main():
    jobs = real_jobs()
    jobs.sort(key=lambda job: int(pd.read_csv(Path(job["input"]) / "filter_summary.csv")["valid_pairs"].iloc[0]))
    for job in jobs:
        run_dataset(job)
    from scripts.summarize_method_comparison import summarize
    summarize()
    subprocess.run(
        [sys.executable, str(PROJECT / "scripts/plot_fine_k_comparison.py")],
        cwd=PROJECT, env={**os.environ, "FINE_K_SOURCE": "real"}, check=True,
    )


if __name__ == "__main__":
    main()
