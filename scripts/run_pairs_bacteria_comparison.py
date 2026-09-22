#!/usr/bin/env python3
"""Add the joint BH pairs + collapsed-bacteria method to cached comparisons."""

import hashlib
import multiprocessing as mp
from pathlib import Path
import sys
from time import perf_counter

from scipy import sparse

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from scripts.run_method_comparison import (
    K_VALUES, MAX_REAL_WORKERS, MAX_SYNTHETIC_WORKERS, OUTPUT_ROOT, SCREEN_FDR,
    SOURCE, TOTAL_CPUS,
    add_job_columns, add_job_values, bh_family, collapsed_average,
    evaluate_ranked, feature_table, is_complete, load_job, output_path,
    permuted_targets, real_jobs, rsp_row, run_signature, save_method,
    split_and_normalize, sqrt_sparse, synthetic_jobs,
)


METHOD = "bh_pairs_bacteria"
SIDE_VERSION = 1


def side_signature(job, y, files):
    base = run_signature(job, y, files)
    return hashlib.sha256(f"{base}:{METHOD}:{SIDE_VERSION}".encode()).hexdigest()


def run_job(job):
    started = perf_counter()
    matrix, y, pair_bacteria, _, files = load_job(job)
    destination = output_path(job) / METHOD
    signature = side_signature(job, y, files)
    if is_complete(destination, signature):
        print(f"Cached: {job['source']} {job['dataset']} seed={job['seed']}", flush=True)
        return

    train_index, test_index, mask, _, threshold, train, test = split_and_normalize(matrix, y)
    y_train, y_test = y[train_index], y[test_index]
    retained_bacteria = pair_bacteria[mask]
    pair_train, pair_test = sqrt_sparse(train), sqrt_sparse(test)
    bacteria_train, bacteria_ids = collapsed_average(
        train, retained_bacteria, int(pair_bacteria.max()) + 1
    )
    bacteria_test, test_ids = collapsed_average(
        test, retained_bacteria, int(pair_bacteria.max()) + 1
    )
    if not (bacteria_ids == test_ids).all():
        raise RuntimeError("Train/test bacteria columns changed")

    source_train = sparse.hstack([pair_train, bacteria_train], format="csr")
    source_test = sparse.hstack([pair_test, bacteria_test], format="csr")
    targets = permuted_targets(y_train, f"{job['source']}:{job['dataset']}:{job['seed']}")
    result = bh_family(source_train, targets)
    selected_order = result["order"][result["q"][result["order"]] <= SCREEN_FDR]
    maximum = max(K_VALUES)
    pair_ids = mask.nonzero()[0]
    levels = ["pair"] * len(pair_ids) + ["bacterium"] * len(bacteria_ids)
    ids = list(pair_ids) + list(bacteria_ids)
    ranking = feature_table(selected_order, levels, ids, result, maximum)
    selected = feature_table(selected_order, levels, ids, result)
    ranked_train = source_train[:, selected_order[:maximum]]
    ranked_test = source_test[:, selected_order[:maximum]]
    counts = selected["level"].value_counts()
    metrics, predictions = evaluate_ranked(
        METHOD, ranked_train, ranked_test, ranking, y_train, y_test,
        {"bacteria": len(bacteria_ids), "gene": 0, "pair": len(pair_ids)},
        {"bacteria": int(counts.get("bacterium", 0)), "gene": 0,
         "pair": int(counts.get("pair", 0))},
    )
    add_job_columns(metrics, job)
    add_job_columns(predictions, job)
    save_method(
        destination, signature, ranking, ranked_train, ranked_test, metrics,
        predictions, add_job_values(rsp_row(METHOD, result["discoveries"]), job),
        selected, perf_counter() - started,
    )
    print(
        f"Completed {job['source']} {job['dataset']} seed={job['seed']}: "
        f"retained pairs={len(pair_ids):,}, bacteria={len(bacteria_ids)}, "
        f"threshold={threshold}, elapsed={perf_counter() - started:.1f}s",
        flush=True,
    )


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE in {"all", "real"}:
        run_jobs(real_jobs(), MAX_REAL_WORKERS, "real pairs+bacteria")
    if SOURCE in {"all", "synthetic"}:
        run_jobs(synthetic_jobs(), MAX_SYNTHETIC_WORKERS, "synthetic pairs+bacteria")
    from scripts.summarize_method_comparison import summarize
    summarize()


def run_jobs(jobs, maximum_workers, label):
    workers = min(maximum_workers, len(jobs), TOTAL_CPUS)
    print(f"Running {len(jobs)} {label} jobs with {workers} workers", flush=True)
    with mp.get_context("spawn").Pool(workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs)


if __name__ == "__main__":
    main()
