#!/usr/bin/env python3
"""Compare marginal and residual-pair signal with held-out prediction."""

from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


PROJECT = Path("/home/kapachy/Microbium/tensor_feature_selection")
DATASET = os.environ.get("DATASET", "WirbelJ_2018")
INPUT = PROJECT / "outputs/preprocessing" / DATASET
OUTPUT = PROJECT / "outputs/feature_selection" / DATASET
PREVALENCE_FRACTION = 0.05
MINIMUM_SAMPLES = 10
METRIC = "correlation"
SIMILARITY_THRESHOLD = 0.93 if METRIC == "correlation" else 0.90
JACCARD_THRESHOLD = 0.80
MAX_CLUSTER_ROUNDS = 2
SCREEN_FDR = 0.10
PAIR_FDR = 0.10
MAX_SELECTED_PER_LEVEL = 200
HOLDOUT_TEST_FRACTION = 0.20
INNER_FOLDS = 3
HOLDOUT_SEED = 42
MODEL_SEED = 42

sys.path.insert(0, str(PROJECT))
from research_utils import (  # noqa: E402
    aggregate_columns,
    benjamini_hochberg,
    mann_whitney_scores,
    marginal_matrix,
    normalize_rows,
    prevalence_mask,
    read_lines,
    residual_correlation_scores,
    similarity_coclustering,
    sqrt_sparse,
)


def new_model():
    return make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=MODEL_SEED,
            solver="liblinear",
        ),
    )


def predict_model(train_x, train_y, test_x):
    if train_x.shape[1] == 0:
        return np.full(test_x.shape[0], np.mean(train_y))
    model = new_model()
    model.fit(train_x, train_y)
    return model.predict_proba(test_x)[:, 1]


def selected_columns(matrix, y, limit=MAX_SELECTED_PER_LEVEL):
    if matrix.shape[1] == 0:
        return np.array([], dtype=int), None
    p, q, effect, auc = mann_whitney_scores(matrix, y)
    selected = np.flatnonzero(q <= SCREEN_FDR)
    if len(selected) > limit:
        selected = selected[np.argsort(q[selected])[:limit]]
    table = pd.DataFrame({"column": np.arange(matrix.shape[1]), "p": p, "q": q, "effect": effect, "auc": auc})
    return selected, table


def inner_splits(y):
    splitter = StratifiedKFold(INNER_FOLDS, shuffle=True, random_state=MODEL_SEED)
    return list(splitter.split(np.zeros(len(y)), y))


def outcome_residuals(features, y, splits):
    probability = np.zeros(len(y))
    for train, test in splits:
        probability[test] = predict_model(features[train], y[train], features[test])
    return y - probability


def residual_pair_features(pair_train, pair_test, marg_train, marg_test, y):
    """Screen against outcome residuals, then residualize shortlisted pairs."""
    splits = inner_splits(y)
    outcome_residual = outcome_residuals(marg_train, y, splits)
    _, _, pair_q = residual_correlation_scores(sqrt_sparse(pair_train), outcome_residual)
    candidates = np.flatnonzero(pair_q <= PAIR_FDR)
    if len(candidates) > MAX_SELECTED_PER_LEVEL:
        candidates = candidates[np.argsort(pair_q[candidates])[:MAX_SELECTED_PER_LEVEL]]
    if not len(candidates):
        return sparse.csr_matrix((len(y), 0)), sparse.csr_matrix((pair_test.shape[0], 0)), candidates

    raw_train = sqrt_sparse(pair_train[:, candidates]).toarray()
    raw_test = sqrt_sparse(pair_test[:, candidates]).toarray()
    predicted = np.zeros_like(raw_train)
    if marg_train.shape[1]:
        for train, test in splits:
            model = Ridge(alpha=1.0)
            model.fit(marg_train[train], raw_train[train])
            predicted[test] = model.predict(marg_train[test])
    else:
        predicted[:] = raw_train.mean(axis=0)
    pair_residual = raw_train - predicted
    _, p_values, _ = residual_correlation_scores(pair_residual, outcome_residual)
    q_values = benjamini_hochberg(p_values)
    kept = np.flatnonzero(q_values <= PAIR_FDR)
    candidates = candidates[kept]
    if not len(candidates):
        return sparse.csr_matrix((len(y), 0)), sparse.csr_matrix((pair_test.shape[0], 0)), candidates

    raw_train = sqrt_sparse(pair_train[:, candidates]).toarray()
    raw_test = sqrt_sparse(pair_test[:, candidates]).toarray()
    if marg_train.shape[1]:
        model = Ridge(alpha=1.0)
        model.fit(marg_train, raw_train)
        residual_train = raw_train - model.predict(marg_train)
        residual_test = raw_test - model.predict(marg_test)
    else:
        mean = raw_train.mean(axis=0)
        residual_train, residual_test = raw_train - mean, raw_test - mean
    return sparse.csr_matrix(residual_train), sparse.csr_matrix(residual_test), candidates


def holdout_representations(x_train, x_test, y_train, edge_bacteria, edge_gene):
    """Learn filtering, clusters, and supervised screens from the training split."""
    mask, _, threshold = prevalence_mask(x_train, PREVALENCE_FRACTION, MINIMUM_SAMPLES)
    train = normalize_rows(x_train[:, mask])
    test = normalize_rows(x_test[:, mask])
    pair_bacteria, pair_gene = edge_bacteria[mask], edge_gene[mask]
    active_bacteria, local_bacteria = np.unique(pair_bacteria, return_inverse=True)
    active_genes, local_genes = np.unique(pair_gene, return_inverse=True)

    bacteria_labels_active, gene_labels_active = None, None
    active_bacteria_check, bacteria_labels_active, active_genes_check, gene_labels_active = similarity_coclustering(
        train,
        pair_bacteria,
        pair_gene,
        int(edge_bacteria.max()) + 1,
        int(edge_gene.max()) + 1,
        SIMILARITY_THRESHOLD,
        JACCARD_THRESHOLD,
        METRIC,
        MAX_CLUSTER_ROUNDS,
        verbose=True,
    )
    if not np.array_equal(active_bacteria, active_bacteria_check) or not np.array_equal(active_genes, active_genes_check):
        raise RuntimeError("Active entity order changed during clustering")

    bacteria_train = marginal_matrix(train, local_bacteria, len(active_bacteria))
    bacteria_test = marginal_matrix(test, local_bacteria, len(active_bacteria))
    genes_train = marginal_matrix(train, local_genes, len(active_genes))
    genes_test = marginal_matrix(test, local_genes, len(active_genes))
    bacteria_cluster_train, bacteria_labels = aggregate_columns(bacteria_train, bacteria_labels_active)
    bacteria_cluster_test, _ = aggregate_columns(bacteria_test, bacteria_labels_active)
    gene_cluster_train, gene_labels = aggregate_columns(genes_train, gene_labels_active)
    gene_cluster_test, _ = aggregate_columns(genes_test, gene_labels_active)

    selected_b, _ = selected_columns(sqrt_sparse(bacteria_cluster_train), y_train)
    selected_g, _ = selected_columns(sqrt_sparse(gene_cluster_train), y_train)
    selected_raw_b, _ = selected_columns(sqrt_sparse(bacteria_train), y_train)
    selected_raw_g, _ = selected_columns(sqrt_sparse(genes_train), y_train)

    b_train, b_test = sqrt_sparse(bacteria_cluster_train[:, selected_b]), sqrt_sparse(bacteria_cluster_test[:, selected_b])
    g_train, g_test = sqrt_sparse(gene_cluster_train[:, selected_g]), sqrt_sparse(gene_cluster_test[:, selected_g])
    clustered_train = sparse.hstack([b_train, g_train], format="csr")
    clustered_test = sparse.hstack([b_test, g_test], format="csr")
    unclustered_train = sparse.hstack([
        sqrt_sparse(bacteria_train[:, selected_raw_b]),
        sqrt_sparse(genes_train[:, selected_raw_g]),
    ], format="csr")
    unclustered_test = sparse.hstack([
        sqrt_sparse(bacteria_test[:, selected_raw_b]),
        sqrt_sparse(genes_test[:, selected_raw_g]),
    ], format="csr")

    details = {
        "threshold": threshold,
        "pair_mask": mask,
        "pair_train": train,
        "pair_test": test,
        "pair_bacteria": pair_bacteria,
        "pair_gene": pair_gene,
        "active_bacteria": active_bacteria,
        "active_genes": active_genes,
        "bacteria_labels": bacteria_labels,
        "gene_labels": gene_labels,
        "selected_bacteria_clusters": selected_b,
        "selected_gene_clusters": selected_g,
        "selected_raw_bacteria": selected_raw_b,
        "selected_raw_genes": selected_raw_g,
    }
    representations = {
        "bacteria_clusters": (b_train, b_test),
        "gene_clusters": (g_train, g_test),
        "marginal_clusters": (clustered_train, clustered_test),
        "unclustered_marginals": (unclustered_train, unclustered_test),
    }
    return representations, details


# Step 1: load the valid pairs. Every later choice is training-split-specific.
matrix = sparse.load_npz(INPUT / "X_valid.npz").tocsr()
samples = pd.read_csv(INPUT / "samples.csv")
edges = np.load(INPUT / "edges.npz")
edge_bacteria, edge_gene = edges["pair_bacteria"], edges["pair_gene"]
bacteria = read_lines(INPUT / "bacteria.txt.gz")
genes = read_lines(INPUT / "genes.txt.gz")
y = samples["y"].to_numpy(dtype=int)
if set(np.unique(y)) != {0, 1}:
    raise ValueError("Expected binary labels 0 and 1")

# Step 2: make one fixed stratified holdout, shared by every representation.
metric_rows, selection_rows, fold_info_rows = [], [], []
splitter = StratifiedShuffleSplit(
    n_splits=1,
    test_size=HOLDOUT_TEST_FRACTION,
    random_state=HOLDOUT_SEED,
)
train_index, test_index = next(splitter.split(matrix, y))
representations, details = holdout_representations(
    matrix[train_index], matrix[test_index], y[train_index],
    edge_bacteria, edge_gene,
)
probabilities = {"intercept": np.full(len(test_index), np.mean(y[train_index]))}
for mode, (train_x, test_x) in representations.items():
    probabilities[mode] = predict_model(train_x, y[train_index], test_x)

marg_train, marg_test = representations["marginal_clusters"]
pair_train, pair_test, selected_pairs = residual_pair_features(
    details["pair_train"], details["pair_test"], marg_train, marg_test,
    y[train_index],
)
combined_train = sparse.hstack([marg_train, pair_train], format="csr")
combined_test = sparse.hstack([marg_test, pair_test], format="csr")
probabilities["marginal_plus_pairs"] = predict_model(
    combined_train, y[train_index], combined_test
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
    "unclustered_marginals": (
        len(details["selected_raw_bacteria"]),
        len(details["selected_raw_genes"]),
        0,
    ),
    "marginal_plus_pairs": (
        representations["bacteria_clusters"][0].shape[1],
        representations["gene_clusters"][0].shape[1],
        len(selected_pairs),
    ),
}
expected_widths = {
    "intercept": 0,
    **{mode: train_x.shape[1] for mode, (train_x, _) in representations.items()},
    "marginal_plus_pairs": combined_train.shape[1],
}
for mode, (n_bacteria, n_genes, n_pairs) in feature_counts.items():
    total = n_bacteria + n_genes + n_pairs
    if total != expected_widths[mode]:
        raise RuntimeError(f"Feature count does not match {mode} model width")
    fold_info_rows.append({
        "evaluation": "fixed_holdout",
        "repeat": 1,
        "fold": 1,
        "fold_seed": HOLDOUT_SEED,
        "test_fraction": HOLDOUT_TEST_FRACTION,
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
print(
    f"Fixed holdout: {len(train_index)} train, {len(test_index)} test; "
    f"{details['pair_train'].shape[1]:,} retained pairs; model features "
    + ", ".join(f"{mode}={sum(counts)}" for mode, counts in feature_counts.items()),
    flush=True,
)

for cluster in details["selected_bacteria_clusters"]:
    members = details["active_bacteria"][details["bacteria_labels"] == cluster]
    for member in members:
        selection_rows.append({"repeat": 1, "fold": 1, "level": "bacterium", "feature": bacteria[member]})
for cluster in details["selected_gene_clusters"]:
    members = details["active_genes"][details["gene_labels"] == cluster]
    for member in members:
        selection_rows.append({"repeat": 1, "fold": 1, "level": "gene", "feature": genes[member]})
for pair in selected_pairs:
    selection_rows.append({
        "repeat": 1, "fold": 1, "level": "pair",
        "feature": f"{genes[details['pair_gene'][pair]]}|{bacteria[details['pair_bacteria'][pair]]}",
    })

for mode, probability in probabilities.items():
    metric_rows.append({
        "evaluation": "fixed_holdout",
        "repeat": 1,
        "fold_seed": HOLDOUT_SEED,
        "test_fraction": HOLDOUT_TEST_FRACTION,
        "mode": mode,
        "roc_auc": roc_auc_score(y[test_index], probability),
        "average_precision": average_precision_score(y[test_index], probability),
    })

# Step 3: summarize holdout prediction and the selected features.
metrics = pd.DataFrame(metric_rows)
fold_model_info = pd.DataFrame(fold_info_rows)
summary = metrics.merge(
    fold_model_info[[
        "mode", "bacteria_features", "gene_features",
        "pair_features", "total_features",
    ]],
    on="mode",
    validate="one_to_one",
)
selections = pd.DataFrame(selection_rows, columns=["repeat", "fold", "level", "feature"])
if len(selections):
    stability = selections.drop_duplicates().groupby(["level", "feature"], as_index=False).size()
    stability["selection_frequency"] = stability["size"]
    stability = stability.drop(columns="size").sort_values(
        ["level", "selection_frequency", "feature"], ascending=[True, False, True]
    )
else:
    stability = pd.DataFrame(columns=["level", "feature", "selection_frequency"])

OUTPUT.mkdir(parents=True, exist_ok=True)
metrics.to_csv(OUTPUT / "repeat_metrics.csv", index=False, float_format="%.3f")
summary.to_csv(OUTPUT / "metrics.csv", index=False, float_format="%.3f")
fold_model_info.to_csv(OUTPUT / "fold_model_info.csv", index=False)
stability.to_csv(OUTPUT / "selected_feature_stability.csv", index=False, float_format="%.3f")

print(f"Samples: {len(y)}; positives: {y.sum()}")
print("\nFixed stratified holdout")
print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
print(f"\nSaved results to {OUTPUT}")
