#!/usr/bin/env python3
"""Restartable 80/20 comparison of three MW/BH methods on real datasets."""
from pathlib import Path
import hashlib
import json
import os
import resource
import sys
from time import perf_counter

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ.setdefault(name, '1')
PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import StratifiedShuffleSplit
from research_utils import (normalize_rows, prevalence_mask, collapsed_average, bh_family,
    feature_table, predict_model, cache_signature, file_hash)
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score

REAL_DATASETS = "all"
K_VALUES = [5, 10, 20, 35, 50, 60, 75, 85, 100]
TEST_FRACTION, HOLDOUT_SEED, MODEL_SEED = .20, 42, 42
PREVALENCE_FRACTION, MINIMUM_SAMPLES, SCREEN_FDR = .05, 5, .10
CACHE_VERSION = 7
OUTPUT_ROOT = PROJECT/'outputs/method_comparison'
REAL_LABELS = {
    "HMP_2019_ibdmdb_unique": "IBD", "JieZ_2017": "ACVD", "LiJ_2017": "hypertension",
    "NielsenHB_2014_unique": "IBD", "QinJ_2012": "T2D", "RubelMA_2020": "STH",
    "WirbelJ_2018": "CRC", "ZhuF_2020": "schizophrenia",
}
METHOD_LABELS = {"bh_bacteria": "Bacteria-collapsed mean | MW/BH",
    "bh_genes": "Gene-collapsed mean | MW/BH",
    "bernoulli_correlation_join": "Bernoulli-join mean | MW/BH"}
METHODS = list(METHOD_LABELS)

def real_jobs():
    names = list(REAL_LABELS) if REAL_DATASETS == "all" else (
        [REAL_DATASETS] if isinstance(REAL_DATASETS, str) else list(REAL_DATASETS)
    )
    unknown = sorted(set(names) - set(REAL_LABELS))
    if unknown:
        raise ValueError(f"Unknown real datasets: {unknown}")
    return [{
        "source": "real", "dataset": name, "n": None, "seed": None,
        "scenario": "real", "input": str(PROJECT / "outputs/preprocessing" / name),
    } for name in names]

def output_path(job):
    return OUTPUT_ROOT/'real'/job['dataset']


def load_job(job):
    source = Path(job['input'])
    matrix = sparse.load_npz(source/'X_valid.npz').tocsr()
    samples = pd.read_csv(source/'samples.csv', dtype={'sample_id': str})
    with np.load(source/'edges.npz') as edges:
        pb, pg = edges['pair_bacteria'], edges['pair_gene']
    y = samples.y.to_numpy(dtype=int)
    if (matrix.shape != (len(y), len(pb)) or len(pg) != matrix.shape[1]
            or set(np.unique(y)) != {0, 1} or not np.isfinite(matrix.data).all()
            or np.any(matrix.data < 0) or np.any(np.asarray(matrix.sum(axis=1)).ravel() <= 0)
            or samples.sample_id.isna().any() or samples.sample_id.duplicated().any()
            or ('subject_id' in samples and (samples.subject_id.isna().any() or samples.subject_id.duplicated().any()))
            or not np.issubdtype(pb.dtype, np.integer) or not np.issubdtype(pg.dtype, np.integer)
            or np.any(pb < 0) or np.any(pg < 0)):
        raise ValueError(f'Invalid independent-sample matrix/mappings: {job["dataset"]}')
    files = [source/n for n in ('X_valid.npz', 'edges.npz', 'samples.csv')]
    return matrix, y, pb, pg, files


def run_signature(job, y, files):
    settings = dict(cache_version=CACHE_VERSION, dataset=job['dataset'],
        inputs=[(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in files],
        k=K_VALUES, split=[TEST_FRACTION, HOLDOUT_SEED],
        prevalence=[PREVALENCE_FRACTION, MINIMUM_SAMPLES], bh=SCREEN_FDR,
        model='Training-scaled balanced L2 C=1', model_seed=MODEL_SEED,
        code={str(p):file_hash(p) for p in (Path(__file__), PROJECT/'research_utils.py')})
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    digest.update(np.asarray(y, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def write_completion(path, signature, extra=None):
    path.mkdir(parents=True, exist_ok=True)
    temporary = path / "complete.json.tmp"
    content = {"cache_version": CACHE_VERSION, "signature": signature, **(extra or {})}
    temporary.write_text(json.dumps(content, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path / "complete.json")

def is_complete(path, signature):
    marker = path/'complete.json'
    if not marker.exists():
        return False
    return json.loads(marker.read_text())['signature'] == cache_signature(marker, signature)


def split_and_normalize(matrix, y):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_FRACTION, random_state=HOLDOUT_SEED
    )
    train_index, test_index = next(splitter.split(matrix, y))
    mask, support, threshold = prevalence_mask(
        matrix[train_index], PREVALENCE_FRACTION, MINIMUM_SAMPLES
    )
    train = normalize_rows(matrix[train_index][:, mask])
    test = normalize_rows(matrix[test_index][:, mask])
    return train_index, test_index, mask, support, threshold, train, test


def evaluate_ranked(
    method, train, test, ranking, y_train, y_test, output_counts,
    selected_counts=None, k_values=None, method_label=None,
):
    selected_counts = output_counts if selected_counts is None else selected_counts
    k_values = K_VALUES if k_values is None else k_values
    method_label = METHOD_LABELS[method] if method_label is None else method_label
    rows, predictions = [], []
    score_cache = {}
    for requested_k in k_values:
        effective_k = min(requested_k, train.shape[1])
        if effective_k not in score_cache:
            probability = predict_model(
                train[:, :effective_k], y_train, test[:, :effective_k], MODEL_SEED
            )
            predicted = (probability >= 0.5).astype(int)
            score_cache[effective_k] = (probability, predicted, {
                "auroc": float(roc_auc_score(y_test, probability)),
                "precision": float(precision_score(y_test, predicted, zero_division=0)),
                "recall": float(recall_score(y_test, predicted, zero_division=0)),
                "f1": float(f1_score(y_test, predicted, zero_division=0)),
            })
        probability, predicted, scores = score_cache[effective_k]
        used = ranking.iloc[:effective_k]
        counts = used["level"].value_counts()
        rows.append({
            "method": method, "method_label": method_label,
            "requested_k": requested_k, "effective_k": effective_k,
            "bacteria_features": int(counts.get("bacterium", 0)),
            "gene_features": int(counts.get("gene", 0)),
            "pair_features": int(counts.get("pair", 0)),
            "pair_cluster_features": int(counts.get("pair_cluster", 0)),
            "total_features": effective_k,
            **{f"output_{level}_features": int(output_counts.get(level, 0))
               for level in ("bacteria", "gene", "pair", "pair_cluster")},
            "output_total_features": int(sum(output_counts.values())),
            **{f"selected_{level}_features": int(selected_counts.get(level, 0))
               for level in ("bacteria", "gene", "pair", "pair_cluster")},
            "selected_total_features": int(sum(selected_counts.values())),
            **scores,
        })
        predictions.extend({
            "method": method, "requested_k": requested_k,
            "effective_k": effective_k, "test_position": position,
            "y": int(target), "probability": float(probability[position]),
            "prediction": int(predicted[position]),
        } for position, target in enumerate(y_test))
    return pd.DataFrame(rows), pd.DataFrame(predictions)

def save_method(path, signature, ranking, train, test, metrics, predictions,
                selected, runtime=0., completion_extra=None):
    path.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(path/'ranked_features.csv.gz', index=False)
    sparse.save_npz(path/'ranked_train.npz', train)
    sparse.save_npz(path/'ranked_test.npz', test)
    metrics.to_csv(path/'metrics.csv', index=False, float_format='%.8g')
    predictions.to_csv(path/'predictions.csv.gz', index=False, float_format='%.8g')
    selected.to_csv(path/'selected_features.csv.gz', index=False)
    write_completion(path, signature, dict(runtime_seconds=runtime,
        peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
        **(completion_extra or {})))


def add_job_columns(table, job):
    table.insert(0, "scenario", job["scenario"])
    table.insert(0, "seed", job["seed"])
    table.insert(0, "n", job["n"])
    table.insert(0, "dataset", job["dataset"])
    table.insert(0, "source", job["source"])
    return table


def add_job_values(row, job):
    return {
        "source": job["source"], "dataset": job["dataset"], "n": job["n"],
        "seed": job["seed"], "scenario": job["scenario"], **row,
    }

def run_job(job):
    source = Path(job['input'])
    y = pd.read_csv(source/'samples.csv').y.to_numpy(int)
    signature = run_signature(job, y, [source/n for n in ('X_valid.npz', 'edges.npz', 'samples.csv')])
    destination = output_path(job)
    missing = [m for m in ('bh_bacteria', 'bh_genes') if not is_complete(destination/m, signature)]
    if any((destination/m/'complete.json').exists() for m in missing):
        raise ValueError('Cached baseline inputs/code changed; use a new output directory')
    if not missing:
        print(f'Cached baselines: {job["dataset"]}', flush=True)
        return
    matrix, y, pb, pg, _ = load_job(job)
    a, b, mask, _, cutoff, train, test = split_and_normalize(matrix, y)
    print(f'{job["dataset"]}: samples={len(y)}, classes={np.bincount(y).tolist()}, '
          f'pairs={mask.sum():,}, bacteria={len(np.unique(pb[mask])):,}, genes={len(np.unique(pg[mask])):,}', flush=True)
    for method in missing:
        started = perf_counter()
        entity, level = (pb, 'bacterium') if method == 'bh_bacteria' else (pg, 'gene')
        t, ids = collapsed_average(train, entity[mask], int(entity.max())+1)
        v, other_ids = collapsed_average(test, entity[mask], int(entity.max())+1)
        if not np.array_equal(ids, other_ids): raise ValueError('Marginal columns changed')
        scores = bh_family(t, [y[a]])
        order = scores['order'][scores['q'][scores['order']] <= SCREEN_FDR]
        selected = feature_table(order, np.full(len(ids), level), ids, scores)
        ranking = selected.iloc[:max(K_VALUES)].copy()
        t, v = t[:, order[:max(K_VALUES)]], v[:, order[:max(K_VALUES)]]
        family = 'bacteria' if method == 'bh_bacteria' else 'gene'
        metrics, predictions = evaluate_ranked(method, t, v, ranking, y[a], y[b],
            {family:len(ids)}, {family:len(order)})
        add_job_columns(metrics, job)
        add_job_columns(predictions, job)
        save_method(destination/method, signature, ranking, t, v, metrics, predictions,
                    selected, perf_counter()-started)


def main():
    from scripts.run_bernoulli_correlation_join import run_dataset
    for job in real_jobs():
        run_job(job)
        run_dataset(job)
    from scripts.summarize_method_comparison import summarize
    from scripts.plot_fine_k_comparison import main as fine_plots
    summarize()
    fine_plots()


if __name__ == '__main__':
    main()
