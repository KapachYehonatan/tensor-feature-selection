#!/usr/bin/env python3
"""Compare marginal and residual-pair signal with held-out prediction."""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/home/kapachy/Microbium/tensor_feature_selection")
DATASET = os.environ.get("DATASET", "WirbelJ_2018")
INPUT = PROJECT / "outputs/preprocessing" / DATASET
OUTPUT = PROJECT / "outputs/feature_selection" / DATASET
PREVALENCE_FRACTION = 0.05
MINIMUM_SAMPLES = 5
METRIC = "correlation"
SIMILARITY_THRESHOLD = 0.93 if METRIC == "correlation" else 0.90
JACCARD_THRESHOLD = 0.80
MAX_CLUSTER_ROUNDS = 2
SCREEN_FDR = 0.10
PAIR_CANDIDATES_PER_BLOCK = 2
PAIR_L1_C = 0.033
HOLDOUT_TEST_FRACTION = 0.20
INNER_FOLDS = 3
HOLDOUT_SEED = 42
MODEL_SEED = 42

sys.path.insert(0, str(PROJECT))
from research_utils import (  # noqa: E402
    aggregate_columns,
    grouped_partial_correlation_scores,
    grouped_residual_features,
    mann_whitney_scores,
    marginal_matrix,
    normalize_rows,
    prevalence_mask,
    read_lines,
    similarity_coclustering,
    sqrt_sparse,
)


def new_model(model_seed=MODEL_SEED):
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=model_seed,
            solver="liblinear",
        ),
    )


def predict_model(train_x, train_y, test_x, model_seed=MODEL_SEED):
    if train_x.shape[1] == 0:
        return np.full(test_x.shape[0], np.mean(train_y))
    model = new_model(model_seed)
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def selected_columns(matrix, y, screen_fdr=SCREEN_FDR):
    if matrix.shape[1] == 0:
        return np.array([], dtype=int), None
    p, q, effect, auc = mann_whitney_scores(matrix, y)
    selected = np.flatnonzero(q <= screen_fdr)
    table = pd.DataFrame({
        "column": np.arange(matrix.shape[1]),
        "p": p,
        "q": q,
        "effect": effect,
        "auc": auc,
    })
    return selected, table


def inner_splits(y, inner_folds=INNER_FOLDS, model_seed=MODEL_SEED):
    splitter = StratifiedKFold(inner_folds, shuffle=True, random_state=model_seed)
    return list(splitter.split(np.zeros(len(y)), y))


def outcome_residuals(features, y, splits, model_seed=MODEL_SEED):
    probability = np.zeros(len(y))
    for train, test in splits:
        probability[test] = predict_model(
            features[train], y[train], features[test], model_seed
        )
    return y - probability


def fit_sequential_selector(
    pair_train,
    pair_test,
    pair_blocks,
    marg_train,
    y,
    candidates_per_block=PAIR_CANDIDATES_PER_BLOCK,
    pair_l1_c=PAIR_L1_C,
    inner_folds_count=INNER_FOLDS,
    model_seed=MODEL_SEED,
):
    """Fit the current conditional pair selector and expose its fitted details."""
    if candidates_per_block <= 0 or pair_l1_c <= 0:
        raise ValueError("Pair candidates per block and L1 C must be positive")
    splits = inner_splits(y, inner_folds_count, model_seed)
    outcome_residual = outcome_residuals(marg_train, y, splits, model_seed)
    transformed_train = sqrt_sparse(pair_train)
    transformed_test = sqrt_sparse(pair_test)
    pair_score, _, _ = grouped_partial_correlation_scores(
        transformed_train, outcome_residual, pair_blocks
    )
    candidates = []
    for block in np.unique(pair_blocks):
        members = np.flatnonzero(pair_blocks == block)
        order = np.argsort(-np.abs(pair_score[members]), kind="stable")
        candidates.extend(members[order[:candidates_per_block]])
    candidates = np.asarray(candidates, dtype=int)
    residual_train, residual_test = grouped_residual_features(
        transformed_train, transformed_test, pair_blocks, candidates
    )

    selection_train = sparse.hstack([marg_train, residual_train], format="csr")
    selector = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=pair_l1_c,
            l1_ratio=1,
            class_weight="balanced",
            max_iter=2000,
            random_state=model_seed,
            solver="liblinear",
        ),
    )
    selector.fit(selection_train, y)
    coefficient = selector[-1].coef_.ravel()
    return {
        "candidates": candidates,
        "pair_scores": pair_score[candidates],
        "residual_train": residual_train,
        "residual_test": residual_test,
        "marginal_coefficients": coefficient[:marg_train.shape[1]],
        "pair_coefficients": coefficient[marg_train.shape[1]:],
    }


def residual_pair_features(
    pair_train,
    pair_test,
    pair_blocks,
    marg_train,
    y,
    candidates_per_block=PAIR_CANDIDATES_PER_BLOCK,
    pair_l1_c=PAIR_L1_C,
    inner_folds_count=INNER_FOLDS,
    model_seed=MODEL_SEED,
):
    """Return every residual pair receiving a nonzero L1 coefficient."""
    details = fit_sequential_selector(
        pair_train,
        pair_test,
        pair_blocks,
        marg_train,
        y,
        candidates_per_block,
        pair_l1_c,
        inner_folds_count,
        model_seed,
    )
    selected = np.flatnonzero(details["pair_coefficients"] != 0)
    return (
        details["residual_train"][:, selected],
        details["residual_test"][:, selected],
        details["candidates"][selected],
    )


def cluster_training_representations(
    x_train,
    x_test,
    edge_bacteria,
    edge_gene,
    prevalence_fraction=PREVALENCE_FRACTION,
    minimum_samples=MINIMUM_SAMPLES,
    metric=METRIC,
    similarity_threshold=SIMILARITY_THRESHOLD,
    jaccard_threshold=JACCARD_THRESHOLD,
    max_cluster_rounds=MAX_CLUSTER_ROUNDS,
    verbose=False,
):
    """Learn filtering and co-clusters from training samples only."""
    mask, _, threshold = prevalence_mask(
        x_train, prevalence_fraction, minimum_samples
    )
    train = normalize_rows(x_train[:, mask])
    test = normalize_rows(x_test[:, mask])
    pair_bacteria, pair_gene = edge_bacteria[mask], edge_gene[mask]
    active_bacteria, local_bacteria = np.unique(pair_bacteria, return_inverse=True)
    active_genes, local_genes = np.unique(pair_gene, return_inverse=True)

    clustering = similarity_coclustering(
        train,
        pair_bacteria,
        pair_gene,
        int(edge_bacteria.max()) + 1,
        int(edge_gene.max()) + 1,
        similarity_threshold,
        jaccard_threshold,
        metric,
        max_cluster_rounds,
        return_diagnostics=True,
        verbose=verbose,
    )
    (
        active_bacteria_check,
        bacteria_labels_active,
        active_genes_check,
        gene_labels_active,
        cluster_diagnostics,
    ) = clustering
    if not np.array_equal(active_bacteria, active_bacteria_check) or not np.array_equal(
        active_genes, active_genes_check
    ):
        raise RuntimeError("Active entity order changed during clustering")

    bacteria_train = marginal_matrix(train, local_bacteria, len(active_bacteria))
    bacteria_test = marginal_matrix(test, local_bacteria, len(active_bacteria))
    genes_train = marginal_matrix(train, local_genes, len(active_genes))
    genes_test = marginal_matrix(test, local_genes, len(active_genes))
    bacteria_cluster_train, bacteria_labels = aggregate_columns(
        bacteria_train, bacteria_labels_active
    )
    bacteria_cluster_test, _ = aggregate_columns(
        bacteria_test, bacteria_labels_active
    )
    gene_cluster_train, gene_labels = aggregate_columns(
        genes_train, gene_labels_active
    )
    gene_cluster_test, _ = aggregate_columns(genes_test, gene_labels_active)
    bacteria_cluster = np.full(int(edge_bacteria.max()) + 1, -1, dtype=int)
    gene_cluster = np.full(int(edge_gene.max()) + 1, -1, dtype=int)
    bacteria_cluster[active_bacteria] = bacteria_labels
    gene_cluster[active_genes] = gene_labels
    raw_pair_blocks = (
        bacteria_cluster[pair_bacteria] * len(np.unique(gene_labels))
        + gene_cluster[pair_gene]
    )
    _, pair_blocks = np.unique(raw_pair_blocks, return_inverse=True)

    matrices = {
        "bacteria": (bacteria_train, bacteria_test),
        "genes": (genes_train, genes_test),
        "bacteria_clusters": (bacteria_cluster_train, bacteria_cluster_test),
        "gene_clusters": (gene_cluster_train, gene_cluster_test),
    }
    details = {
        "threshold": threshold,
        "pair_mask": mask,
        "pair_train": train,
        "pair_test": test,
        "pair_bacteria": pair_bacteria,
        "pair_gene": pair_gene,
        "pair_blocks": pair_blocks,
        "active_bacteria": active_bacteria,
        "active_genes": active_genes,
        "bacteria_labels": bacteria_labels,
        "gene_labels": gene_labels,
        "cluster_diagnostics": cluster_diagnostics,
    }
    return matrices, details


def holdout_representations(
    x_train,
    x_test,
    y_train,
    edge_bacteria,
    edge_gene,
    prevalence_fraction=PREVALENCE_FRACTION,
    minimum_samples=MINIMUM_SAMPLES,
    metric=METRIC,
    similarity_threshold=SIMILARITY_THRESHOLD,
    jaccard_threshold=JACCARD_THRESHOLD,
    max_cluster_rounds=MAX_CLUSTER_ROUNDS,
    screen_fdr=SCREEN_FDR,
    verbose=False,
):
    """Learn clusters and supervised marginal screens from the training split."""
    matrices, details = cluster_training_representations(
        x_train,
        x_test,
        edge_bacteria,
        edge_gene,
        prevalence_fraction,
        minimum_samples,
        metric,
        similarity_threshold,
        jaccard_threshold,
        max_cluster_rounds,
        verbose,
    )
    bacteria_cluster_train, bacteria_cluster_test = matrices["bacteria_clusters"]
    gene_cluster_train, gene_cluster_test = matrices["gene_clusters"]

    selected_b, _ = selected_columns(
        sqrt_sparse(bacteria_cluster_train), y_train, screen_fdr
    )
    selected_g, _ = selected_columns(
        sqrt_sparse(gene_cluster_train), y_train, screen_fdr
    )
    b_train = sqrt_sparse(bacteria_cluster_train[:, selected_b])
    b_test = sqrt_sparse(bacteria_cluster_test[:, selected_b])
    g_train = sqrt_sparse(gene_cluster_train[:, selected_g])
    g_test = sqrt_sparse(gene_cluster_test[:, selected_g])
    clustered_train = sparse.hstack([b_train, g_train], format="csr")
    clustered_test = sparse.hstack([b_test, g_test], format="csr")
    details.update({
        "selected_bacteria_clusters": selected_b,
        "selected_gene_clusters": selected_g,
    })
    representations = {
        "bacteria_clusters": (b_train, b_test),
        "gene_clusters": (g_train, g_test),
        "marginal_clusters": (clustered_train, clustered_test),
    }
    return representations, details


def run_holdout_selection(
    matrix,
    y,
    edge_bacteria,
    edge_gene,
    prevalence_fraction=PREVALENCE_FRACTION,
    minimum_samples=MINIMUM_SAMPLES,
    metric=METRIC,
    similarity_threshold=SIMILARITY_THRESHOLD,
    jaccard_threshold=JACCARD_THRESHOLD,
    max_cluster_rounds=MAX_CLUSTER_ROUNDS,
    screen_fdr=SCREEN_FDR,
    pair_candidates_per_block=PAIR_CANDIDATES_PER_BLOCK,
    pair_l1_c=PAIR_L1_C,
    test_fraction=HOLDOUT_TEST_FRACTION,
    inner_folds_count=INNER_FOLDS,
    holdout_seed=HOLDOUT_SEED,
    model_seed=MODEL_SEED,
    verbose=False,
):
    """Run complete training-only selection and fixed-holdout evaluation."""
    matrix = sparse.csr_matrix(matrix)
    y = np.asarray(y, dtype=int)
    edge_bacteria = np.asarray(edge_bacteria, dtype=np.int64)
    edge_gene = np.asarray(edge_gene, dtype=np.int64)
    if matrix.shape[0] != len(y) or matrix.shape[1] != len(edge_bacteria):
        raise ValueError("Matrix, target, and pair mappings do not align")
    if len(edge_gene) != matrix.shape[1] or set(np.unique(y)) != {0, 1}:
        raise ValueError("Expected aligned pair mappings and binary labels 0 and 1")

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=test_fraction, random_state=holdout_seed
    )
    train_index, test_index = next(splitter.split(matrix, y))
    representations, details = holdout_representations(
        matrix[train_index],
        matrix[test_index],
        y[train_index],
        edge_bacteria,
        edge_gene,
        prevalence_fraction,
        minimum_samples,
        metric,
        similarity_threshold,
        jaccard_threshold,
        max_cluster_rounds,
        screen_fdr,
        verbose,
    )

    probabilities = {"intercept": np.full(len(test_index), np.mean(y[train_index]))}
    for mode, (train_x, test_x) in representations.items():
        probabilities[mode] = predict_model(
            train_x, y[train_index], test_x, model_seed
        )

    marg_train, marg_test = representations["marginal_clusters"]
    pair_train, pair_test, selected_pairs = residual_pair_features(
        details["pair_train"],
        details["pair_test"],
        details["pair_blocks"],
        marg_train,
        y[train_index],
        pair_candidates_per_block,
        pair_l1_c,
        inner_folds_count,
        model_seed,
    )
    combined_train = sparse.hstack([marg_train, pair_train], format="csr")
    combined_test = sparse.hstack([marg_test, pair_test], format="csr")
    if len(selected_pairs):
        probabilities["pairs_only"] = predict_model(
            pair_train, y[train_index], pair_test, model_seed
        )
    probabilities["marginal_plus_pairs"] = predict_model(
        combined_train, y[train_index], combined_test, model_seed
    )

    feature_counts = {
        "intercept": (0, 0, 0),
        "bacteria_clusters": (representations["bacteria_clusters"][0].shape[1], 0, 0),
        "gene_clusters": (0, representations["gene_clusters"][0].shape[1], 0),
        "marginal_clusters": (
            representations["bacteria_clusters"][0].shape[1],
            representations["gene_clusters"][0].shape[1],
            0,
        ),
    }
    if len(selected_pairs):
        feature_counts["pairs_only"] = (0, 0, len(selected_pairs))
    feature_counts["marginal_plus_pairs"] = (
        representations["bacteria_clusters"][0].shape[1],
        representations["gene_clusters"][0].shape[1],
        len(selected_pairs),
    )
    expected_widths = {
        "intercept": 0,
        **{mode: train_x.shape[1] for mode, (train_x, _) in representations.items()},
    }
    if len(selected_pairs):
        expected_widths["pairs_only"] = pair_train.shape[1]
    expected_widths["marginal_plus_pairs"] = combined_train.shape[1]
    fold_rows = []
    for mode, (n_bacteria, n_genes, n_pairs) in feature_counts.items():
        total = n_bacteria + n_genes + n_pairs
        if total != expected_widths[mode]:
            raise RuntimeError(f"Feature count does not match {mode} model width")
        fold_rows.append({
            "evaluation": "fixed_holdout",
            "repeat": 1,
            "fold": 1,
            "fold_seed": holdout_seed,
            "test_fraction": test_fraction,
            "mode": mode,
            "train_samples": len(train_index),
            "test_samples": len(test_index),
            "train_positive": int(y[train_index].sum()),
            "test_positive": int(y[test_index].sum()),
            "prevalence_threshold": details["threshold"],
            "retained_pairs": details["pair_train"].shape[1],
            "active_bacteria": len(details["active_bacteria"]),
            "active_genes": len(details["active_genes"]),
            "bacteria_clusters": len(np.unique(details["bacteria_labels"])),
            "gene_clusters": len(np.unique(details["gene_labels"])),
            "bacteria_features": n_bacteria,
            "gene_features": n_genes,
            "pair_features": n_pairs,
            "total_features": total,
        })

    metrics = pd.DataFrame([
        {
            "evaluation": "fixed_holdout",
            "repeat": 1,
            "fold_seed": holdout_seed,
            "test_fraction": test_fraction,
            "mode": mode,
            "roc_auc": roc_auc_score(y[test_index], probability),
            "average_precision": average_precision_score(y[test_index], probability),
        }
        for mode, probability in probabilities.items()
    ])
    fold_model_info = pd.DataFrame(fold_rows)
    summary = metrics.merge(
        fold_model_info[[
            "mode",
            "bacteria_features",
            "gene_features",
            "pair_features",
            "total_features",
        ]],
        on="mode",
        validate="one_to_one",
    )
    selected_bacteria = details["active_bacteria"][np.isin(
        details["bacteria_labels"], details["selected_bacteria_clusters"]
    )]
    selected_genes = details["active_genes"][np.isin(
        details["gene_labels"], details["selected_gene_clusters"]
    )]
    selected_pair_indices = np.flatnonzero(details["pair_mask"])[selected_pairs]
    details["selected_pairs"] = selected_pairs

    if verbose:
        print(
            f"Fixed holdout: {len(train_index)} train, {len(test_index)} test; "
            f"{details['pair_train'].shape[1]:,} retained pairs; model features "
            + ", ".join(
                f"{mode}={sum(counts)}" for mode, counts in feature_counts.items()
            ),
            flush=True,
        )
    return {
        "train_index": train_index,
        "test_index": test_index,
        "metrics": metrics,
        "fold_model_info": fold_model_info,
        "summary": summary,
        "probabilities": probabilities,
        "details": details,
        "selected_bacteria": selected_bacteria,
        "selected_genes": selected_genes,
        "selected_pairs": selected_pair_indices,
    }


def main():
    matrix = sparse.load_npz(INPUT / "X_valid.npz").tocsr()
    samples = pd.read_csv(INPUT / "samples.csv")
    edges = np.load(INPUT / "edges.npz")
    edge_bacteria, edge_gene = edges["pair_bacteria"], edges["pair_gene"]
    bacteria = read_lines(INPUT / "bacteria.txt.gz")
    genes = read_lines(INPUT / "genes.txt.gz")
    y = samples["y"].to_numpy(dtype=int)

    result = run_holdout_selection(
        matrix, y, edge_bacteria, edge_gene, verbose=True
    )
    selection_rows = []
    for member in result["selected_bacteria"]:
        selection_rows.append({
            "repeat": 1,
            "fold": 1,
            "level": "bacterium",
            "feature": bacteria[member],
        })
    for member in result["selected_genes"]:
        selection_rows.append({
            "repeat": 1,
            "fold": 1,
            "level": "gene",
            "feature": genes[member],
        })
    for pair in result["selected_pairs"]:
        selection_rows.append({
            "repeat": 1,
            "fold": 1,
            "level": "pair",
            "feature": f"{genes[edge_gene[pair]]}|{bacteria[edge_bacteria[pair]]}",
        })

    selections = pd.DataFrame(
        selection_rows, columns=["repeat", "fold", "level", "feature"]
    )
    if len(selections):
        stability = (
            selections.drop_duplicates()
            .groupby(["level", "feature"], as_index=False)
            .size()
        )
        stability["selection_frequency"] = stability["size"]
        stability = stability.drop(columns="size").sort_values(
            ["level", "selection_frequency", "feature"],
            ascending=[True, False, True],
        )
    else:
        stability = pd.DataFrame(
            columns=["level", "feature", "selection_frequency"]
        )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    result["metrics"].to_csv(
        OUTPUT / "repeat_metrics.csv", index=False, float_format="%.3f"
    )
    result["summary"].to_csv(
        OUTPUT / "metrics.csv", index=False, float_format="%.3f"
    )
    result["fold_model_info"].to_csv(OUTPUT / "fold_model_info.csv", index=False)
    stability.to_csv(
        OUTPUT / "selected_feature_stability.csv", index=False, float_format="%.3f"
    )

    print(f"Samples: {len(y)}; positives: {y.sum()}")
    print("\nFixed stratified holdout")
    print(
        result["summary"].to_string(
            index=False, float_format=lambda value: f"{value:.3f}"
        )
    )
    print(f"\nSaved results to {OUTPUT}")


if __name__ == "__main__":
    main()
