#!/usr/bin/env python3
"""Build descriptive bacteria and gene clusters from strong similarities."""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import sparse


PROJECT = Path("/home/kapachy/Microbium/tensor_feature_selection")
DATASET = "WirbelJ_2018"
INPUT = PROJECT / "outputs/preprocessing" / DATASET
OUTPUT = PROJECT / "outputs/clustering" / DATASET
PREVALENCE_FRACTION = 0.05
MINIMUM_SAMPLES = 5
METRIC = "correlation"  # Change to "bray_curtis" for the one sensitivity run.
SIMILARITY_THRESHOLD = 0.95 if METRIC == "correlation" else 0.90
JACCARD_THRESHOLD = 0.80
MAX_ROUNDS = 4

sys.path.insert(0, str(PROJECT))
from research_utils import (  # noqa: E402
    marginal_matrix,
    normalize_rows,
    prevalence_mask,
    read_lines,
    similarity_coclustering,
)


def cluster_table(ids, active, labels, marginal):
    abundance = np.asarray(marginal.sum(axis=0)).ravel()
    rows = []
    for cluster in np.unique(labels):
        members = np.flatnonzero(labels == cluster)
        representative = members[np.argmax(abundance[members])]
        for member in members:
            rows.append(
                {
                    "cluster": int(cluster),
                    "entity": ids[active[member]],
                    "representative": member == representative,
                    "cluster_size": len(members),
                    "total_abundance": abundance[member],
                }
            )
    return pd.DataFrame(rows)


# Step 1: load the prepared sparse pairs and apply the configured prevalence rule.
matrix = sparse.load_npz(INPUT / "X_valid.npz").tocsr()
samples = pd.read_csv(INPUT / "samples.csv")
edges = np.load(INPUT / "edges.npz")
pair_gene = edges["pair_gene"]
pair_bacteria = edges["pair_bacteria"]
genes = read_lines(INPUT / "genes.txt.gz")
bacteria = read_lines(INPUT / "bacteria.txt.gz")
mask, support, threshold = prevalence_mask(
    matrix, PREVALENCE_FRACTION, MINIMUM_SAMPLES
)
filtered = normalize_rows(matrix[:, mask])
pair_gene, pair_bacteria = pair_gene[mask], pair_bacteria[mask]

# Step 2: alternate high-threshold similarity clustering across both dimensions.
active_bacteria, bacteria_labels, active_genes, gene_labels = similarity_coclustering(
    filtered,
    pair_bacteria,
    pair_gene,
    len(bacteria),
    len(genes),
    SIMILARITY_THRESHOLD,
    JACCARD_THRESHOLD,
    METRIC,
    MAX_ROUNDS,
    verbose=True,
)

# Step 3: choose the most abundant member as a readable cluster representative.
local_bacteria = np.searchsorted(active_bacteria, pair_bacteria)
local_genes = np.searchsorted(active_genes, pair_gene)
bacteria_marginal = marginal_matrix(filtered, local_bacteria, len(active_bacteria))
gene_marginal = marginal_matrix(filtered, local_genes, len(active_genes))
bacteria_table = cluster_table(
    bacteria, active_bacteria, bacteria_labels, bacteria_marginal
)
gene_table = cluster_table(genes, active_genes, gene_labels, gene_marginal)

# Step 4: save only the filter mask and readable cluster membership tables.
OUTPUT.mkdir(parents=True, exist_ok=True)
np.save(OUTPUT / "pair_mask.npy", mask)
bacteria_table.to_csv(OUTPUT / "bacteria_clusters.csv", index=False, float_format="%.3f")
gene_table.to_csv(OUTPUT / "gene_clusters.csv", index=False, float_format="%.3f")
pd.DataFrame(
    [{
        "dataset": DATASET,
        "metric": METRIC,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "jaccard_threshold": JACCARD_THRESHOLD,
        "sample_support_threshold": threshold,
        "input_pairs": matrix.shape[1],
        "retained_pairs": int(mask.sum()),
        "active_bacteria": len(active_bacteria),
        "bacteria_clusters": bacteria_table["cluster"].nunique(),
        "active_genes": len(active_genes),
        "gene_clusters": gene_table["cluster"].nunique(),
    }]
).to_csv(OUTPUT / "cluster_summary.csv", index=False, float_format="%.3f")

print(
    f"Pairs: {matrix.shape[1]:,} -> {mask.sum():,}; "
    f"sample threshold: {threshold}"
)
print(
    f"Bacteria: {len(active_bacteria):,} -> {bacteria_table['cluster'].nunique():,} clusters; "
    f"genes: {len(active_genes):,} -> {gene_table['cluster'].nunique():,} clusters"
)
print(f"Saved clusters to {OUTPUT}")
