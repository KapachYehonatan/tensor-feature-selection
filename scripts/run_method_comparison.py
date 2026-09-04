#!/usr/bin/env python3
"""Run the sparse Top-K feature-selection comparison with restartable outputs."""

from pathlib import Path
import hashlib
import json
import multiprocessing as mp
import os
import resource
import sys
from time import perf_counter


PROJECT = Path(__file__).resolve().parents[1]
SOURCE = os.environ.get("COMPARISON_SOURCE", "all")  # all, real, or synthetic
REAL_DATASETS = "all"
SYNTHETIC_PRESET = os.environ.get("SYNTHETIC_PRESET", "main")
K_VALUES = [5, 10, 20, 35, 50, 60, 75, 85, 100]
TEST_FRACTION = 0.20
HOLDOUT_SEED = 42
MODEL_SEED = 42
PREVALENCE_FRACTION = 0.05
MINIMUM_SAMPLES = 5
METRIC = "correlation"
SIMILARITY_THRESHOLD = 0.93
JACCARD_THRESHOLD = 0.80
MAX_CLUSTER_ROUNDS = 4
SCREEN_FDR = 0.10
RSP_FDR = 0.05
RSP_BETA = 0.05
RSP_PERMUTATIONS = 5
RSP_SEED = 2026
MANN_WHITNEY_CHUNK = 2000
TOTAL_CPUS = int(os.environ.get("COMPARISON_CPUS", "24"))
MAX_REAL_WORKERS = int(os.environ.get("COMPARISON_REAL_WORKERS", "3"))
MAX_SYNTHETIC_WORKERS = int(os.environ.get("COMPARISON_SYNTHETIC_WORKERS", "5"))
CACHE_VERSION = 6
OUTPUT_ROOT = PROJECT / "outputs/method_comparison"

REAL_LABELS = {
    "HMP_2019_ibdmdb_unique": "IBD",
    "JieZ_2017": "ACVD",
    "LiJ_2017": "hypertension",
    "NielsenHB_2014_unique": "IBD",
    "QinJ_2012": "T2D",
    "RubelMA_2020": "STH",
    "WirbelJ_2018": "CRC",
    "ZhuF_2020": "schizophrenia",
}
METHOD_LABELS = {
    "bh_pairs": "BH Top-K pairs",
    "bh_bacteria": "BH Top-K bacteria-collapsed",
    "bh_genes": "BH Top-K gene-collapsed",
    "clustered_marginals": "Marginal co-clustering Top-K",
    "clustered_plus_pairs": "Marginal co-clustering + partial-BH pairs",
}
METHODS = list(METHOD_LABELS)

if SOURCE not in {"all", "real", "synthetic"}:
    raise ValueError("COMPARISON_SOURCE must be all, real, or synthetic")
if min(TOTAL_CPUS, MAX_REAL_WORKERS, MAX_SYNTHETIC_WORKERS) < 1:
    raise ValueError("Comparison CPU and worker settings must be positive")

# Keep nested numerical threads within the requested total CPU budget.
THREADS_PER_WORKER = max(
    1, TOTAL_CPUS // max(MAX_REAL_WORKERS, MAX_SYNTHETIC_WORKERS)
)
for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS",
):
    os.environ.setdefault(variable, str(THREADS_PER_WORKER))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse, stats  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedShuffleSplit  # noqa: E402

sys.path.insert(0, str(PROJECT))
from feature_selection.run_feature_selection import (  # noqa: E402
    cluster_training_representations,
    pair_bh_scores,
    predict_model,
)
from research_utils import (  # noqa: E402
    benjamini_hochberg,
    marginal_matrix,
    normalize_rows,
    prevalence_mask,
    sqrt_sparse,
)
from scripts.generate_synthetic import load_configuration, tensor_to_csr  # noqa: E402


def synthetic_jobs():
    configuration, _, preset = load_configuration(preset_name=SYNTHETIC_PRESET)
    root = PROJECT / configuration["paths"]["data_root"] / SYNTHETIC_PRESET
    jobs = []
    for n in map(int, preset["sample_sizes"]):
        for seed in map(int, preset["seeds"]):
            dataset = root / f"n_{n}" / f"seed_{seed}" / "full_signals"
            if not (dataset / "target.npy").exists():
                raise FileNotFoundError(f"Missing synthetic full_signals dataset: {dataset}")
            jobs.append({
                "source": "synthetic", "dataset": f"n_{n}", "n": n,
                "seed": seed, "scenario": "full_signals", "input": str(dataset),
            })
    return jobs


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
    if job["source"] == "real":
        return OUTPUT_ROOT / "real" / job["dataset"]
    return (
        OUTPUT_ROOT / "synthetic" / job["dataset"]
        / f"seed_{job['seed']}" / "full_signals"
    )


def load_job(job):
    source = Path(job["input"])
    if job["source"] == "real":
        matrix = sparse.load_npz(source / "X_valid.npz").tocsr()
        samples = pd.read_csv(source / "samples.csv")
        with np.load(source / "edges.npz") as edges:
            pair_bacteria = edges["pair_bacteria"]
            pair_gene = edges["pair_gene"]
        y = samples["y"].to_numpy(dtype=int)
        files = [source / "X_valid.npz", source / "edges.npz", source / "samples.csv"]
    else:
        matrix, pair_bacteria, pair_gene, _ = tensor_to_csr(source)
        y = np.load(source / "target.npy").astype(int)
        files = [source / "tensor.npz", source / "target.npy", source / "metadata.json"]
    if matrix.shape != (len(y), len(pair_bacteria)) or len(pair_gene) != matrix.shape[1]:
        raise ValueError(f"Unaligned matrix, target, or pair mappings for {job['dataset']}")
    if set(np.unique(y)) != {0, 1}:
        raise ValueError(f"Expected both binary classes for {job['dataset']}")
    return matrix, y, np.asarray(pair_bacteria), np.asarray(pair_gene), files


def run_signature(job, y, files):
    file_state = []
    for path in files:
        stat = Path(path).stat()
        file_state.append((str(path), stat.st_size, stat.st_mtime_ns))
    settings = {
        "cache_version": CACHE_VERSION,
        "job": job,
        "files": file_state,
        "k": K_VALUES,
        "test_fraction": TEST_FRACTION,
        "holdout_seed": HOLDOUT_SEED,
        "model_seed": MODEL_SEED,
        "prevalence": [PREVALENCE_FRACTION, MINIMUM_SAMPLES],
        "clustering": [METRIC, SIMILARITY_THRESHOLD, JACCARD_THRESHOLD, MAX_CLUSTER_ROUNDS],
        "selection": [SCREEN_FDR, "partial_correlation_bh"],
        "rsp": [RSP_FDR, RSP_BETA, RSP_PERMUTATIONS, RSP_SEED],
    }
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
    marker = path / "complete.json"
    if not marker.exists():
        return False
    try:
        return json.loads(marker.read_text(encoding="utf-8"))["signature"] == signature
    except (OSError, KeyError, json.JSONDecodeError):
        return False


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


def collapsed_average(matrix, entity, total_entities):
    """Average retained pair columns belonging to each active entity."""
    counts = np.bincount(entity, minlength=total_entities)
    active = np.flatnonzero(counts)
    collapsed = marginal_matrix(matrix, entity, total_entities)[:, active]
    collapsed = collapsed @ sparse.diags(1.0 / counts[active])
    return sqrt_sparse(collapsed), active


def permuted_targets(y, dataset_key):
    dataset_seed = int(hashlib.sha256(dataset_key.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng([RSP_SEED, dataset_seed])
    return [np.asarray(y)] + [rng.permutation(y) for _ in range(RSP_PERMUTATIONS)]


def mann_whitney_many(matrix, targets, chunk_size=MANN_WHITNEY_CHUNK):
    """Test real and permuted labels while each sparse block is dense once."""
    matrix = sparse.csc_matrix(matrix)
    p_values = [np.ones(matrix.shape[1]) for _ in targets]
    real_effect = np.zeros(matrix.shape[1])
    for start in range(0, matrix.shape[1], chunk_size):
        end = min(start + chunk_size, matrix.shape[1])
        values = matrix[:, start:end].toarray()
        for target_number, target in enumerate(targets):
            positive, negative = target == 1, target == 0
            result = stats.mannwhitneyu(
                values[positive], values[negative], axis=0,
                alternative="two-sided", method="asymptotic",
            )
            p_values[target_number][start:end] = np.asarray(result.pvalue)
            if target_number == 0:
                denominator = positive.sum() * negative.sum()
                real_effect[start:end] = 2 * np.asarray(result.statistic) / denominator - 1
    return p_values, real_effect


def bh_family(matrix, targets):
    p_values, effect = mann_whitney_many(matrix, targets)
    real_q = benjamini_hochberg(p_values[0])
    columns = np.arange(matrix.shape[1])
    order = np.lexsort((columns, p_values[0], real_q))
    discoveries = [int(np.sum(benjamini_hochberg(values) <= RSP_FDR)) for values in p_values]
    return {
        "order": order,
        "p": p_values[0],
        "q": real_q,
        "effect": effect,
        "discoveries": discoveries,
        "p_values": p_values,
    }


def rsp_row(method, discoveries):
    real = int(discoveries[0])
    shuffled = np.asarray(discoveries[1:], dtype=float)
    sp = float(shuffled.mean())
    denominator = RSP_BETA * real + sp
    score = 0.0 if denominator == 0 else (RSP_BETA * real - sp) / denominator
    return {
        "method": method, "rp": real, "sp": sp, "rsp": score,
        "beta": RSP_BETA, "bh_fdr": RSP_FDR,
        **{f"permutation_{i + 1}": int(value) for i, value in enumerate(shuffled)},
    }


def feature_table(order, levels, ids, scores, maximum=None):
    if maximum is not None:
        order = order[:maximum]
    return pd.DataFrame({
        "rank": np.arange(1, len(order) + 1),
        "level": np.asarray(levels)[order],
        "feature_id": np.asarray(ids)[order],
        "p": np.asarray(scores.get("p", np.full(len(levels), np.nan)))[order],
        "q": np.asarray(scores.get("q", np.full(len(levels), np.nan)))[order],
        "effect": np.asarray(scores.get("effect", np.full(len(levels), np.nan)))[order],
    })


def evaluate_ranked(
    method, train, test, ranking, y_train, y_test, output_counts,
    selected_counts=None,
):
    selected_counts = output_counts if selected_counts is None else selected_counts
    rows, predictions = [], []
    score_cache = {}
    for requested_k in K_VALUES:
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
            "method": method, "method_label": METHOD_LABELS[method],
            "requested_k": requested_k, "effective_k": effective_k,
            "bacteria_features": int(counts.get("bacterium", 0)),
            "gene_features": int(counts.get("gene", 0)),
            "pair_features": int(counts.get("pair", 0)),
            "total_features": effective_k,
            **{f"output_{level}_features": int(output_counts.get(level, 0))
               for level in ("bacteria", "gene", "pair")},
            "output_total_features": int(sum(output_counts.values())),
            **{f"selected_{level}_features": int(selected_counts.get(level, 0))
               for level in ("bacteria", "gene", "pair")},
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


def save_method(path, signature, ranking, train, test, metrics, predictions, rsp=None, selected=None, runtime=0.0):
    path.mkdir(parents=True, exist_ok=True)
    ranking.to_csv(path / "ranked_features.csv.gz", index=False, compression="gzip")
    sparse.save_npz(path / "ranked_train.npz", train, compressed=True)
    sparse.save_npz(path / "ranked_test.npz", test, compressed=True)
    metrics.to_csv(path / "metrics.csv", index=False, float_format="%.8g")
    predictions.to_csv(path / "predictions.csv.gz", index=False, compression="gzip")
    if rsp is not None:
        pd.DataFrame([rsp]).to_csv(path / "rsp.csv", index=False, float_format="%.8g")
    else:
        (path / "rsp.csv").unlink(missing_ok=True)
    if selected is not None:
        selected.to_csv(path / "selected_features.csv.gz", index=False, compression="gzip")
    else:
        (path / "selected_features.csv.gz").unlink(missing_ok=True)
    write_completion(path, signature, {
        "runtime_seconds": runtime,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })


def cached_clusters(matrix, y, pair_bacteria, pair_gene, train_index, test_index, path, signature):
    if is_complete(path, signature):
        matrices = {
            key: (
                sparse.load_npz(path / f"{key}_train.npz"),
                sparse.load_npz(path / f"{key}_test.npz"),
            )
            for key in ("bacteria_clusters", "gene_clusters")
        }
        with np.load(path / "details.npz") as saved:
            details = {key: saved[key] for key in saved.files}
        details["pair_train"] = normalize_rows(matrix[train_index][:, details["pair_mask"]])
        details["pair_test"] = normalize_rows(matrix[test_index][:, details["pair_mask"]])
        return matrices, details

    started = perf_counter()
    matrices, details = cluster_training_representations(
        matrix[train_index], matrix[test_index], pair_bacteria, pair_gene,
        PREVALENCE_FRACTION, MINIMUM_SAMPLES, METRIC, SIMILARITY_THRESHOLD,
        JACCARD_THRESHOLD, MAX_CLUSTER_ROUNDS, False,
    )
    path.mkdir(parents=True, exist_ok=True)
    for key in ("bacteria_clusters", "gene_clusters"):
        sparse.save_npz(path / f"{key}_train.npz", matrices[key][0], compressed=True)
        sparse.save_npz(path / f"{key}_test.npz", matrices[key][1], compressed=True)
    np.savez_compressed(
        path / "details.npz",
        pair_mask=details["pair_mask"], pair_bacteria=details["pair_bacteria"],
        pair_gene=details["pair_gene"], pair_blocks=details["pair_blocks"],
        active_bacteria=details["active_bacteria"], active_genes=details["active_genes"],
        bacteria_labels=details["bacteria_labels"], gene_labels=details["gene_labels"],
    )
    write_completion(path, signature, {
        "runtime_seconds": perf_counter() - started,
        "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    })
    return matrices, details


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
    started = perf_counter()
    matrix, y, pair_bacteria, pair_gene, files = load_job(job)
    destination = output_path(job)
    signature = run_signature(job, y, files)
    missing = [method for method in METHODS if not is_complete(destination / method, signature)]
    if not missing:
        print(f"Cached: {job['source']} {job['dataset']} seed={job['seed']}", flush=True)
        return

    train_index, test_index, mask, _, threshold, train, test = split_and_normalize(matrix, y)
    y_train, y_test = y[train_index], y[test_index]
    retained_bacteria, retained_gene = pair_bacteria[mask], pair_gene[mask]
    original_pairs = np.flatnonzero(mask)
    targets = permuted_targets(y_train, f"{job['source']}:{job['dataset']}:{job['seed']}")
    maximum = max(K_VALUES)

    simple_spaces = {}
    if "bh_pairs" in missing:
        simple_spaces["bh_pairs"] = (
            sqrt_sparse(train), sqrt_sparse(test),
            np.full(train.shape[1], "pair"), original_pairs,
        )
    if "bh_bacteria" in missing:
        train_b, ids_b = collapsed_average(train, retained_bacteria, int(pair_bacteria.max()) + 1)
        test_b, ids_b_test = collapsed_average(test, retained_bacteria, int(pair_bacteria.max()) + 1)
        if not np.array_equal(ids_b, ids_b_test):
            raise RuntimeError("Train/test bacteria columns changed")
        simple_spaces["bh_bacteria"] = (train_b, test_b, np.full(len(ids_b), "bacterium"), ids_b)
    if "bh_genes" in missing:
        train_g, ids_g = collapsed_average(train, retained_gene, int(pair_gene.max()) + 1)
        test_g, ids_g_test = collapsed_average(test, retained_gene, int(pair_gene.max()) + 1)
        if not np.array_equal(ids_g, ids_g_test):
            raise RuntimeError("Train/test gene columns changed")
        simple_spaces["bh_genes"] = (train_g, test_g, np.full(len(ids_g), "gene"), ids_g)

    for method, (source_train, source_test, levels, ids) in simple_spaces.items():
        method_started = perf_counter()
        result = bh_family(source_train, targets)
        selected_order = result["order"][result["q"][result["order"]] <= SCREEN_FDR]
        ranking = feature_table(selected_order, levels, ids, result, maximum)
        selected = feature_table(selected_order, levels, ids, result)
        ranked_train = source_train[:, selected_order[:maximum]]
        ranked_test = source_test[:, selected_order[:maximum]]
        output_counts = {
            "bacteria": int(np.sum(levels == "bacterium")),
            "gene": int(np.sum(levels == "gene")),
            "pair": int(np.sum(levels == "pair")),
        }
        discoveries = len(selected_order)
        metrics, predictions = evaluate_ranked(
            method, ranked_train, ranked_test, ranking, y_train, y_test, output_counts,
            {
                "bacteria": discoveries if method == "bh_bacteria" else 0,
                "gene": discoveries if method == "bh_genes" else 0,
                "pair": discoveries if method == "bh_pairs" else 0,
            },
        )
        rsp = add_job_values(rsp_row(method, result["discoveries"]), job)
        add_job_columns(metrics, job)
        add_job_columns(predictions, job)
        save_method(
            destination / method, signature, ranking, ranked_train, ranked_test,
            metrics, predictions, rsp, selected, perf_counter() - method_started,
        )
    simple_spaces.clear()

    if any(method in missing for method in ("clustered_marginals", "clustered_plus_pairs")):
        matrices, details = cached_clusters(
            matrix, y, pair_bacteria, pair_gene, train_index, test_index,
            destination / "common_clusters", signature,
        )
        bacteria_train = sqrt_sparse(matrices["bacteria_clusters"][0])
        bacteria_test = sqrt_sparse(matrices["bacteria_clusters"][1])
        gene_train = sqrt_sparse(matrices["gene_clusters"][0])
        gene_test = sqrt_sparse(matrices["gene_clusters"][1])
        bacteria_bh = bh_family(bacteria_train, targets)
        gene_bh = bh_family(gene_train, targets)
        levels = np.concatenate([
            np.full(bacteria_train.shape[1], "bacterium"),
            np.full(gene_train.shape[1], "gene"),
        ])
        ids = np.concatenate([
            np.arange(bacteria_train.shape[1]), np.arange(gene_train.shape[1])
        ])
        combined_scores = {
            key: np.concatenate([bacteria_bh[key], gene_bh[key]])
            for key in ("p", "q", "effect")
        }
        type_order = np.concatenate([
            np.zeros(bacteria_train.shape[1], dtype=int),
            np.ones(gene_train.shape[1], dtype=int),
        ])
        combined_order = np.lexsort(
            (ids, type_order, combined_scores["p"], combined_scores["q"])
        )
        selected_marginal_order = combined_order[
            combined_scores["q"][combined_order] <= SCREEN_FDR
        ]
        combined_train = sparse.hstack([bacteria_train, gene_train], format="csr")
        combined_test = sparse.hstack([bacteria_test, gene_test], format="csr")
        marginal_train = combined_train[:, selected_marginal_order]
        marginal_test = combined_test[:, selected_marginal_order]
        marginal_counts = {
            "bacteria": int(np.sum(levels[selected_marginal_order] == "bacterium")),
            "gene": int(np.sum(levels[selected_marginal_order] == "gene")),
            "pair": 0,
        }
        output_counts = {
            "bacteria": bacteria_train.shape[1],
            "gene": gene_train.shape[1],
            "pair": 0,
        }

        if "clustered_marginals" in missing:
            method_started = perf_counter()
            ranking = feature_table(
                selected_marginal_order, levels, ids, combined_scores, maximum
            )
            selected = feature_table(
                selected_marginal_order, levels, ids, combined_scores
            )
            ranked_train = marginal_train[:, :maximum]
            ranked_test = marginal_test[:, :maximum]
            metrics, predictions = evaluate_ranked(
                "clustered_marginals", ranked_train, ranked_test, ranking,
                y_train, y_test, output_counts, marginal_counts,
            )
            discoveries = np.asarray(bacteria_bh["discoveries"]) + np.asarray(gene_bh["discoveries"])
            add_job_columns(metrics, job)
            add_job_columns(predictions, job)
            save_method(
                destination / "clustered_marginals", signature, ranking,
                ranked_train, ranked_test, metrics, predictions,
                add_job_values(rsp_row("clustered_marginals", discoveries), job), selected,
                perf_counter() - method_started,
            )

        if "clustered_plus_pairs" in missing:
            method_started = perf_counter()
            transformed_pair_train = sqrt_sparse(details["pair_train"])
            transformed_pair_test = sqrt_sparse(details["pair_test"])
            pair_result = pair_bh_scores(
                transformed_pair_train, details["pair_blocks"],
                marginal_train, y_train,
            )
            pair_selected = np.flatnonzero(pair_result["q"] <= SCREEN_FDR)
            all_levels = np.concatenate([
                levels[selected_marginal_order], np.full(len(pair_selected), "pair")
            ])
            all_ids = np.concatenate([
                ids[selected_marginal_order], original_pairs[pair_selected]
            ])
            scores = {
                key: np.concatenate([
                    combined_scores[key][selected_marginal_order], pair_result[key][pair_selected]
                ])
                for key in ("p", "q", "effect")
            }
            type_code = np.array([
                {"bacterium": 0, "gene": 1, "pair": 2}[level]
                for level in all_levels
            ])
            order = np.lexsort((all_ids, type_code, scores["p"], scores["q"]))
            selected = feature_table(order, all_levels, all_ids, scores)
            ranking = selected.iloc[:maximum].copy()
            selected_train = sparse.hstack([
                marginal_train, transformed_pair_train[:, pair_selected]
            ], format="csr")
            selected_test = sparse.hstack([
                marginal_test, transformed_pair_test[:, pair_selected]
            ], format="csr")
            ranked_train = selected_train[:, order[:maximum]]
            ranked_test = selected_test[:, order[:maximum]]
            combined_output_counts = {
                **output_counts, "pair": details["pair_train"].shape[1],
            }
            selected_counts = {
                **marginal_counts, "pair": len(pair_selected),
            }
            metrics, predictions = evaluate_ranked(
                "clustered_plus_pairs", ranked_train, ranked_test, ranking,
                y_train, y_test, combined_output_counts, selected_counts,
            )

            pair_discoveries = [int(np.sum(pair_result["q"] <= RSP_FDR))]
            for target_number, target in enumerate(targets[1:], start=1):
                bacteria_q = benjamini_hochberg(bacteria_bh["p_values"][target_number])
                gene_q = benjamini_hochberg(gene_bh["p_values"][target_number])
                permuted_marginal = sparse.hstack([
                    bacteria_train[:, bacteria_q <= SCREEN_FDR],
                    gene_train[:, gene_q <= SCREEN_FDR],
                ], format="csr")
                permuted_pair = pair_bh_scores(
                    transformed_pair_train, details["pair_blocks"],
                    permuted_marginal, target,
                )
                pair_discoveries.append(int(np.sum(permuted_pair["q"] <= RSP_FDR)))
            discoveries = (
                np.asarray(bacteria_bh["discoveries"])
                + np.asarray(gene_bh["discoveries"])
                + np.asarray(pair_discoveries)
            )
            add_job_columns(metrics, job)
            add_job_columns(predictions, job)
            save_method(
                destination / "clustered_plus_pairs", signature, ranking,
                ranked_train, ranked_test, metrics, predictions,
                add_job_values(rsp_row("clustered_plus_pairs", discoveries), job),
                selected, perf_counter() - method_started,
            )

    print(
        f"Completed {job['source']} {job['dataset']} seed={job['seed']}: "
        f"samples={len(y)}, pairs={matrix.shape[1]:,}, retained={mask.sum():,}, "
        f"threshold={threshold}, elapsed={perf_counter() - started:.1f}s",
        flush=True,
    )


def run_jobs(jobs, maximum_workers, label):
    if not jobs:
        return
    workers = min(maximum_workers, len(jobs), TOTAL_CPUS)
    print(
        f"Running {len(jobs)} {label} jobs with {workers} workers; "
        f"total CPU budget={TOTAL_CPUS}, native threads/worker={THREADS_PER_WORKER}",
        flush=True,
    )
    with mp.get_context("spawn").Pool(processes=workers, maxtasksperchild=1) as pool:
        pool.map(run_job, jobs)


def main():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    if SOURCE in {"all", "real"}:
        run_jobs(real_jobs(), MAX_REAL_WORKERS, "real")
    if SOURCE in {"all", "synthetic"}:
        run_jobs(synthetic_jobs(), MAX_SYNTHETIC_WORKERS, "synthetic full_signals")
    from scripts.summarize_method_comparison import summarize
    summarize()


if __name__ == "__main__":
    main()
